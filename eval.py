import json
import time
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm
from metrics import compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ImageDataset(Dataset):
    def __init__(self, image_paths, processor, texts):
        self.image_paths = image_paths
        self.processor = processor
        self.texts = texts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt")
        # Remove the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, image.size, os.path.basename(image_path)

def collate_fn(batch):
    inputs = {
        'pixel_values': [],
        'input_ids': None,
        'attention_mask': None
    }
    sizes = []
    image_names = []
     
    for item in batch:
        inputs['pixel_values'].append(item[0]['pixel_values'])
        if inputs['input_ids'] is None:
            inputs['input_ids'] = item[0]['input_ids']
            inputs['attention_mask'] = item[0]['attention_mask']
        else: 
            # Compare with subsequent items
            assert torch.equal(item[0]['input_ids'], inputs['input_ids']), f"input_ids mismatch at item {idx}"
            assert torch.equal(item[0]['attention_mask'], inputs['attention_mask']), f"attention_mask mismatch at item {idx}"
        sizes.append(item[1])
        image_names.append(item[2])
    
    # Stack pixel_values
    inputs['pixel_values'] = torch.stack(inputs['pixel_values'])

    return inputs, torch.tensor(sizes), image_names


def evaluate_model(model, dataloader, device, processor, name_to_category_id, texts):
    model.eval()
    all_detections = []

    with torch.no_grad():
        for batch, sizes, image_names in tqdm(dataloader, desc="Processing batches"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=sizes, threshold=0.3)
                
            for result, image_name in zip(results, image_names):
                image_id = int(image_name.split('.')[0])
                boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
                
                for box, score, label in zip(boxes, scores, labels):
                    box = [round(i, 2) for i in box.tolist()]
                    x1, y1, x2, y2 = tuple(box)
                    
                    if x1 <= x2 and y1 <= y2:
                        category_id = name_to_category_id.get(texts[0][label], None)
                        
                        all_detections.append({
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'score': score.item()
                        })

    return all_detections

def main():
    # Load annotations and set up paths
    model_id = 1
    gt_file = 'datasets/val_updated.json'
    image_dir = '/home/c3-0/datasets/XView/train_images'
    model_weights = 'google/owlv2-base-patch16-ensemble'    
    output_dir = f'outputs/{model_id}'
    pred_file = 'pred.json'
    eval_results_file = 'eval_results.json'
    eval_detailed_file = 'eval_detailed.txt'
    
    pred_file_path = os.path.join(output_dir, pred_file)
    
    # Start the timer
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations and set up paths
    annotations = json.load(open(gt_file))
    image_paths = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
    
    # Set up category mappings
    category_lookup = {category['id']: category['name'] for category in annotations['categories']}
    unique_category_names = list(set(category_lookup.values()))    
    name_to_category_id = {name: id for id, name in category_lookup.items()}
    
    # Set up model and processor
    processor = Owlv2Processor.from_pretrained(model_weights)
    model = Owlv2ForObjectDetection.from_pretrained(model_weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    texts = [unique_category_names]
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, processor, texts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Evaluate model
    all_detections = evaluate_model(model, dataloader, device, processor, name_to_category_id, texts)
    
    # Save results
    with open(pred_file_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
    
    print(f"Predictions saved to {pred_file_path}")
    
    # End the timer
    end_time = time.time()
    
    # Calculate total time taken
    total_time = end_time - start_time
    print(f"Total processing time [Predictions successfully saved to {pred_file}]: {total_time:.2f} seconds")
    
    compute_metrics(model_id, gt_file, output_dir, pred_file, eval_results_file, eval_detailed_file)

if __name__ == '__main__':
    main()
