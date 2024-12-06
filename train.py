
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Owlv2Processor
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from datetime import datetime
from utils import generate_experiment_name

import torch
from datasets import XViewRandomCrop, collate_fn
from model import XViewDetector
from losses import loss_function
from functools import partial


# Save checkpoint
def save_checkpoint(state, is_best, checkpoint_dir):
    last_ckpt_path = os.path.join(checkpoint_dir, 'last_epoch.pth')
    best_ckpt_path = os.path.join(checkpoint_dir, 'best_epoch.pth')
    
    # Save last checkpoint
    torch.save(state, last_ckpt_path)
    
    # If this is the best model so far, save it as the best checkpoint
    if is_best:
        torch.save(state, best_ckpt_path)

# Load checkpoint
def load_checkpoint(ckpt_path, model, optimizer, scheduler=None):
    if os.path.isfile(ckpt_path):
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
        return start_epoch, best_val_loss, epochs_since_improvement
    else:
        raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'")

# Training Loop
def train_one_epoch(model, train_loader, processor, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        inputs = processor(text=batch['texts'], images=batch['inputs'], return_tensors="pt")
        batch['labels'] = batch['labels'].to(device)
        batch['boxes'] = batch['boxes'].to(device)

        optimizer.zero_grad()
        output = model(**{k:v.to(device) for k, v in inputs.items()})
        loss = criterion(output, batch)[0]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Training... [{i}]/[{len(train_loader)}] Loss: {running_loss/(i+1):.2f}")

    return running_loss / len(train_loader)

# Validation Loop
def validate(model, val_loader, processor, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            inputs = processor(text=batch['texts'], images=batch['inputs'], return_tensors="pt")
            batch['labels'] = batch['labels'].to(device)
            batch['boxes'] = batch['boxes'].to(device)

            output = model(**{k:v.to(device) for k, v in inputs.items()})
            loss = criterion(output, batch)[0]
            val_loss += loss.item()
            if i % 10 == 0:
                print(f"Validating... [{i}]/[{len(val_loader)}] Loss: {val_loss/(i+1):.2f}")

    return val_loss / len(val_loader)

# Main function to handle setup
def setup(rank, args):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # Setup device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Generate unique experiment name
    experiment_name = generate_experiment_name(args.output_dir, base_name="experiment")
    experiment_path = os.path.join(args.output_dir, experiment_name)

    # Create directories for logs and checkpoints
    checkpoint_dir = os.path.join(experiment_path, 'checkpoints')
    log_dir = os.path.join(experiment_path, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer (only rank 0 writes logs)
    if rank != 0:
        global print
        # f = open(os.devnull, "w")
        # sys.stdout = f
        # sys.stderr = f
        def no_print(*args, **kwargs):
            pass
        print = no_print

    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    # Log experiment name
    print(experiment_name)

    # Load dataset and create DataLoader
    train_dataset = XViewRandomCrop(annotations_file=args.train_annotations_file, img_dir=args.img_dir, mode='train')
    val_dataset = XViewRandomCrop(annotations_file=args.val_annotations_file, img_dir=args.img_dir, mode='val')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=args.num_workers)

    processor = Owlv2Processor.from_pretrained(args.model_base)

    # Initialize model and move it to the device
    model = XViewDetector(args.model_base).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = loss_function

    # Initialize scheduler if enabled
    scheduler = None
    if args.use_scheduler:
        if args.scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=args.scheduler_factor,
                                                             patience=args.scheduler_patience,
                                                             verbose=True)
        elif args.scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                  gamma=args.gamma)
        else:
            raise ValueError(f"Unsupported scheduler type: {args.scheduler_type}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        start_epoch, best_val_loss, epochs_since_improvement = load_checkpoint(
            args.ckpt_path, model, optimizer, scheduler)

    # Early Stopping parameters
    patience = args.early_stopping_patience
    if patience > 0:
        enable_early_stopping = True
    else:
        enable_early_stopping = False

    # Training and validation loops
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        # Set epoch for sampler
        train_sampler.set_epoch(epoch)

        # Training
        train_loss = train_one_epoch(model, train_loader, processor, optimizer, criterion, device)

        # Validation
        val_loss = validate(model, val_loader, processor, criterion, device)

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Determine if this is the best validation loss
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early Stopping
        if enable_early_stopping and epochs_since_improvement >= patience:
            if rank == 0:
                print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        if rank == 0:
            print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)

            # Save checkpoint
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_since_improvement': epochs_since_improvement,
            }
            if scheduler:
                checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()
            save_checkpoint(checkpoint_state, is_best, checkpoint_dir)

    # Close the writer
    if rank == 0:
        writer.close()

    # Cleanup
    dist.destroy_process_group()

# Main function to parse arguments and start training
def main():
    parser = argparse.ArgumentParser(description="XView OwlVit2 Finetuning")
    parser.add_argument('--world_size', type=int, default=2, help='number of distributed processes')
    parser.add_argument('--epochs', type=int, default=10, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--img_dir', type=str, default='images/', help='path of image directory')
    parser.add_argument('--train_annotations_file', type=str, default='train.csv', help='path to training annotations.csv')
    parser.add_argument('--val_annotations_file', type=str, default='val.csv', help='path to validation annotations.csv')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='directory to save experiment outputs')
    parser.add_argument('--ckpt_path', type=str, help='path to a checkpoint to resume training from')
    parser.add_argument('--model_base', type=str, default='google/owlv2-base-patch16-ensemble', help='pretrained weights to be used from huggingface')


    # Early Stopping Arguments
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='patience for early stopping (number of epochs with no improvement)')

    # Learning Rate Scheduler Arguments
    parser.add_argument('--use_scheduler', action='store_true',
                        help='whether to use a learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='ReduceLROnPlateau',
                        choices=['ReduceLROnPlateau', 'StepLR'],
                        help='type of learning rate scheduler to use')
    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help='factor by which the learning rate will be reduced')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='number of epochs with no improvement after which learning rate will be reduced (for ReduceLROnPlateau)')
    parser.add_argument('--step_size', type=int, default=7,
                        help='period of learning rate decay (for StepLR)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay (for StepLR)')

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get the number of available GPUs and launch training processes
    args.world_size = torch.cuda.device_count()

    if args.world_size < 1:
        raise ValueError("No GPUs available for training.")
    mp.spawn(setup, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    main()
