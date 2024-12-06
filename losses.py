# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Losses."""
import torch
import torch.nn.functional as F
from typing import Union, Optional
import box_utils
from hungarian import hungarian_matcher
import functools

EPS = 1e-6


# def sigmoid_cost(
#     logit: Union[jnp.ndarray, float],
#     *,
#     focal_loss: bool = False,
#     focal_alpha: Optional[float] = None,
#     focal_gamma: Optional[float] = None
# ) -> Union[jnp.ndarray, float]:
#   """Computes the classification cost.

#   Relevant code:
#   https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/matcher.py#L76

#   Args:
#     logit: Sigmoid classification logit(s).
#     focal_loss: Whether to apply focal loss for classification cost.
#     focal_alpha: Alpha scaling factor for focal loss.
#     focal_gamma: Gamma scaling factor for focal loss.

#   Returns:
#     Classification cost.
#   """
#   neg_cost_class = -jax.nn.log_sigmoid(-logit)
#   pos_cost_class = -jax.nn.log_sigmoid(logit)
#   if focal_loss:
#     neg_cost_class *= (1 - focal_alpha) * jax.nn.sigmoid(logit)**focal_gamma
#     pos_cost_class *= focal_alpha * jax.nn.sigmoid(-logit)**focal_gamma
#   return pos_cost_class - neg_cost_class  # [B, N, C]

def sigmoid_cost(
    logit: Union[torch.Tensor, float],
    *,
    focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: Optional[float] = None
) -> Union[torch.Tensor, float]:
    """Computes the classification cost.

    Args:
        logit: Sigmoid classification logit(s).
        focal_loss: Whether to apply focal loss for classification cost.
        focal_alpha: Alpha scaling factor for focal loss.
        focal_gamma: Gamma scaling factor for focal loss.

    Returns:
        Classification cost.
    """
    # Compute the negative and positive costs using log_sigmoid.
    neg_cost_class = -F.logsigmoid(-logit)
    pos_cost_class = -F.logsigmoid(logit)

    if focal_loss:
        # Apply focal loss scaling factors.
        neg_cost_class *= (1 - focal_alpha) * torch.sigmoid(logit)**focal_gamma
        pos_cost_class *= focal_alpha * torch.sigmoid(-logit)**focal_gamma
    
    # Return the difference between positive and negative costs.
    return pos_cost_class - neg_cost_class  # [B, N, C]

# def compute_cost(
#     *,
#     tgt_labels: jnp.ndarray,
#     out_logits: jnp.ndarray,
#     tgt_bbox: jnp.ndarray,
#     out_bbox: jnp.ndarray,
#     class_loss_coef: float,
#     bbox_loss_coef: jnp.ndarray,
#     giou_loss_coef: jnp.ndarray,
#     focal_loss: bool = False,
#     focal_alpha: Optional[float] = None,
#     focal_gamma: Optional[float] = None,
# ) -> jnp.ndarray:
#   """Computes cost matrices for a batch of predictions.

#   Relevant code:
#   https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

#   Args:
#     tgt_labels: Class labels of shape [B, M, C] (one/multi-hot). Note that the
#       labels corresponding to empty bounding boxes are not yet supposed to be
#       filtered out.
#     out_logits: Classification sigmoid logits of shape [B, N, C].
#     tgt_bbox: Target box coordinates of shape [B, M, 4]. Note that the empty
#       bounding boxes are not yet supposed to be filtered out.
#     out_bbox: Predicted box coordinates of shape [B, N, 4]
#     class_loss_coef: Relative weight of classification loss.
#     bbox_loss_coef: Relative weight of bbox loss.
#     giou_loss_coef: Relative weight of giou loss.
#     focal_loss: Whether to apply focal loss for classification cost.
#     focal_alpha: Alpha scaling factor for focal loss.
#     focal_gamma: Gamma scaling factor for focal loss.

#   Returns:
#     A cost matrix [B, N, M].
#     Number of unpadded columns per batch element [B].
#   """
#   if focal_loss and (focal_alpha is None or focal_gamma is None):
#     raise ValueError('For focal loss, focal_alpha and focal_gamma must be set.')

#   # Number of non-padding labels for each of the target instances.
#   n_labels_per_instance = jnp.sum(tgt_labels[..., 1:], axis=-1)
#   mask = n_labels_per_instance > 0  # [B, M]

#   # Make sure padding target is 0 for instances with other labels.
#   tgt_labels = jnp.concatenate(
#       [jnp.expand_dims(~mask, -1), tgt_labels[..., 1:]], axis=-1)

#   cost_class = sigmoid_cost(  # [B, N, C]
#       out_logits,
#       focal_loss=focal_loss,
#       focal_alpha=focal_alpha,
#       focal_gamma=focal_gamma)

#   # Resulting shape is [B, N, M].
#   # Note that we do *not* normalize by the number of per-target instances.
#   cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)

#   cost = class_loss_coef * cost_class

#   diff = jnp.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])  # [B, N, M, 4]
#   cost_bbox = jnp.sum(diff, axis=-1)  # [B, N, M]
#   cost = cost + bbox_loss_coef * cost_bbox

#   cost_giou = -box_utils.generalized_box_iou(
#       box_utils.box_cxcywh_to_xyxy(out_bbox),
#       box_utils.box_cxcywh_to_xyxy(tgt_bbox),
#       all_pairs=True)
#   cost = cost + giou_loss_coef * cost_giou

#   mask = mask[:, None]

#   # Determine mask value dynamically.
#   cost_mask_value = jnp.max(jnp.where(mask, cost, -1e10), axis=(1, 2))
#   # Special case.
#   all_masked = jnp.all(~mask, axis=(1, 2))
#   cost_mask_value = jnp.where(~all_masked, cost_mask_value, 1.0)
#   cost_mask_value = cost_mask_value[:, None, None] * 1.1 + 10.0

#   cost = cost * mask + (1.0 - mask) * cost_mask_value
#   # Guard against NaNs and Infs.
#   cost = jnp.nan_to_num(
#       cost,
#       nan=cost_mask_value,
#       posinf=cost_mask_value,
#       neginf=cost_mask_value)

#   # Compute the number of unpadded columns for each batch element. It is assumed
#   # that all padding is trailing padding.
#   max_num_boxes = tgt_labels.shape[1]
#   n_cols = jnp.where(
#       jnp.max(mask, axis=1),
#       jnp.expand_dims(jnp.arange(1, max_num_boxes + 1), axis=0), 0)
#   n_cols = jnp.max(n_cols, axis=1)
#   return cost, n_cols  # pytype: disable=bad-return-type  # jax-ndarray

@torch.no_grad()
def compute_cost(
    *,
    tgt_labels: torch.Tensor,
    out_logits: torch.Tensor,
    tgt_bbox: torch.Tensor,
    out_bbox: torch.Tensor,
    class_loss_coef: float,
    bbox_loss_coef: torch.Tensor,
    giou_loss_coef: torch.Tensor,
    focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: Optional[float] = None,
) -> torch.Tensor:
    """Computes cost matrices for a batch of predictions.

    Args:
        tgt_labels: Class labels of shape [B, M, C] (one/multi-hot). Note that the
          labels corresponding to empty bounding boxes are not yet supposed to be
          filtered out.
        out_logits: Classification sigmoid logits of shape [B, N, C].
        tgt_bbox: Target box coordinates of shape [B, M, 4]. Note that the empty
          bounding boxes are not yet supposed to be filtered out.
        out_bbox: Predicted box coordinates of shape [B, N, 4]
        class_loss_coef: Relative weight of classification loss.
        bbox_loss_coef: Relative weight of bbox loss.
        giou_loss_coef: Relative weight of giou loss.
        focal_loss: Whether to apply focal loss for classification cost.
        focal_alpha: Alpha scaling factor for focal loss.
        focal_gamma: Gamma scaling factor for focal loss.

    Returns:
        A cost matrix [B, N, M].
        Number of unpadded columns per batch element [B].
    """
    if focal_loss and (focal_alpha is None or focal_gamma is None):
        raise ValueError('For focal loss, focal_alpha and focal_gamma must be set.')

    # Number of non-padding labels for each of the target instances.
    n_labels_per_instance = torch.sum(tgt_labels[..., 1:], dim=-1)
    mask = n_labels_per_instance > 0  # [B, M]

    # Make sure padding target is 0 for instances with other labels.
    tgt_labels = torch.cat([~mask.unsqueeze(-1), tgt_labels[..., 1:]], dim=-1)

    cost_class = sigmoid_cost(
        out_logits,
        focal_loss=focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma)

    # Resulting shape is [B, N, M].
    # Note that we do *not* normalize by the number of per-target instances.
    cost_class = torch.einsum('bnl,bml->bnm', cost_class, tgt_labels)

    cost = class_loss_coef * cost_class

    diff = torch.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])  # [B, N, M, 4]
    cost_bbox = torch.sum(diff, dim=-1)  # [B, N, M]
    cost = cost + bbox_loss_coef * cost_bbox

    cost_giou = -box_utils.generalized_box_iou(
        box_utils.box_cxcywh_to_xyxy(out_bbox),
        box_utils.box_cxcywh_to_xyxy(tgt_bbox),
        all_pairs=True)
    cost = cost + giou_loss_coef * cost_giou

    mask = mask.unsqueeze(1)

    # Determine mask value dynamically.
    cost_mask_value = torch.amax(torch.where(mask, cost, torch.tensor(-1e10, dtype=cost.dtype)), dim=(1, 2))

    # Special case.
    all_masked = torch.all(~mask, dim=(1, 2))
    cost_mask_value = torch.where(~all_masked, cost_mask_value, torch.tensor(1.0, dtype=cost.dtype))
    cost_mask_value = cost_mask_value.unsqueeze(1).unsqueeze(1) * 1.1 + 10.0

    # mask = mask.to(torch.int)
    cost = cost * mask.to(torch.int) + (1.0 - mask.to(torch.int)) * cost_mask_value
    # Guard against NaNs and Infs.
    cost = torch.nan_to_num(cost, nan=cost_mask_value.max(), posinf=cost_mask_value.max(), neginf=cost_mask_value.max())

    # Compute the number of unpadded columns for each batch element. It is assumed
    # that all padding is trailing padding.
    max_num_boxes = tgt_labels.shape[1]
    n_cols = torch.where(
        torch.max(mask, dim=1)[0],
        torch.arange(1, max_num_boxes + 1, device=tgt_labels.device).unsqueeze(0), 0)
    n_cols = torch.max(n_cols, dim=1)[0]
    return cost, n_cols  # pytype: disable=bad-return-type  # jax-ndarray


def matcher(cost, n_cols = None):
    """Implements a matching function.

    Matching functions match predicted detections against ground truth
    detections and return match indices.

    Args:
      cost: Matching cost matrix [B, N, M].
      n_cols: Number of non-padded columns in each cost matrix.

    Returns:
      Matched indices in the form of a list of tuples (src, dst), where
      `src` and `dst` are indices of corresponding predicted and ground truth
      detections. [B, 2, min(N, M)].
    """
    # matcher_fn = functools.partial(hungarian_matcher, n_cols=n_cols)
    matcher_fn = hungarian_matcher
    
    # return matcher_fn(cost.detach())
    device = cost.device
    cost = cost.detach().cpu()

    across_batch = []
    for inp in cost:
        across_batch.append(torch.from_numpy(matcher_fn(inp)).to(device))

    return torch.stack(across_batch).to(torch.int64)
    

def loss_function(outputs, batch):

    device = batch['labels'].device
    label_shape = batch['labels'].shape
    num_classes = label_shape[-1]
    instance = F.one_hot(torch.tensor(0), num_classes).to(device)
    reshape_shape = (1,) * (len(label_shape) - 1) + (num_classes,)
    broadcast_shape = label_shape[:-2] + (1, num_classes)
    instance = torch.broadcast_to(
      torch.reshape(instance, reshape_shape), broadcast_shape
    )
    batch['labels'] = torch.cat([batch['labels'], instance], dim=-2)

    instance = torch.zeros_like(batch['boxes'][..., :1, :])
    batch['boxes'] = torch.cat([batch['boxes'], instance], dim=-2)

    outputs['logits'] = torch.cat([-1e10 * torch.ones(*outputs['logits'].shape[:2], 1).to(device), outputs['logits']], dim=-1) # B x N x C+1 adding the null class dim to output

    cost, n_cols = compute_cost(
        tgt_labels=batch['labels'],
        out_logits=outputs['logits'],
        tgt_bbox=batch['boxes'],
        out_bbox=outputs['pred_boxes'],
        class_loss_coef=1.0,
        bbox_loss_coef=1.0,
        giou_loss_coef=1.0,
        focal_loss=True,
        focal_alpha=0.3,
        focal_gamma=2.0,
        )
    matches = matcher(cost, n_cols)

    if not isinstance(matches, (list, tuple)):
        # Ensure matches come as a sequence.
        matches = [matches]
    
    num_pred = outputs['logits'].shape[-2]

    def pad_matches(match):
        batch_size, _, num_matched = match.shape  # [B, 2, M]
        if num_pred > num_matched:

            def get_unmatched_indices(row, ind):
                # row = row.copy()
                row[ind] = torch.tensor(1, device=device, dtype=row.dtype)
                return torch.topk(torch.logical_not(row).to(torch.int64), k=num_pred - num_matched)

            get_unmatched_indices = torch.vmap(get_unmatched_indices)

            indices = torch.zeros((batch_size, num_pred), dtype=torch.bool).to(device)
            _, indices = get_unmatched_indices(indices, match[:, 0, :])
            indices = indices.unsqueeze(1)

            padding = torch.cat(
              [indices, torch.full(indices.shape, fill_value=num_matched - 1, dtype=torch.long, device=device)],
              dim=1
            )
            return torch.cat([match, padding], dim=-1)
        return match

    matches = [pad_matches(match) for match in matches]

    indices = matches[0]

    loss_dict = {}
    metrics_dict = {}
    losses_and_metrics = ['labels', 'boxes']
    loss_and_metrics_map = {
      'labels': labels_losses_and_metrics, 
      'boxes': boxes_losses_and_metrics
    }
    
    def get_losses_and_metrics(loss, outputs, batch, indices, **kwargs):
        """A convenience wrapper to all the loss_* functions in this class."""
        assert loss in loss_and_metrics_map, f'Unknown loss {loss}.'
        return loss_and_metrics_map[loss](outputs, batch, indices, **kwargs)

    for loss_name in losses_and_metrics:
      loss, metrics = get_losses_and_metrics(loss_name, outputs, batch, indices)
      loss_dict.update(loss)
      metrics_dict.update(metrics)

    # Compute the total loss by combining loss_dict with loss_terms_weights.
    total_loss = []
    for k, v in loss_dict.items():
      # if k in loss_terms_weights:
        # total_loss.append(loss_terms_weights[k] * v)
      total_loss.append(v)
    total_loss = sum(total_loss)

    # Process metrics dictionary to generate final unnormalized metrics.
    metrics = get_metrics(metrics_dict)
    metrics['total_loss'] = (total_loss, 1)
    return total_loss, metrics  # pytype: disable=bad-return-type  # jax-ndarray

def get_metrics(metrics_dict):
    """Arrange loss dictionary into a metrics dictionary."""
    metrics = {}
    # Some metrics don't get scaled, so no need to keep their unscaled version,
    # i.e. those that are not in self.loss_terms_weights.keys()
    for k, v in metrics_dict.items():
      # loss_term = self.loss_terms_weights.get(k)
      loss_term = None
      if loss_term is not None:
        metrics[f'{k}_unscaled'] = v
        metrics[k] = (loss_term * v[0], v[1])
      else:
        metrics[k] = v

    return metrics

def labels_losses_and_metrics(
      outputs,
      batch,
      indices,
      log = True):
    """Classification loss.

    Args:
      outputs: Model predictions. For the purpose of this loss, outputs must
        have key 'pred_logits'. outputs['pred_logits'] is a nd-array of the
        predicted logits of shape [batch-size, num-objects, num-classes].
      batch: Dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, label dict must
        have key 'labels', which the value is an int nd-array of labels with
        shape [batch_size, num_boxes, num_classes + 1]. Since the number of
        boxes (objects) in each example in the batch could be different, the
        input pipeline might add padding boxes to some examples. These padding
        boxes are identified based on their class labels. So if the class label
        is `0`, i.e., a one-hot vector of [1, 0, 0, ..., 0], the box/object is a
        padding object and the loss computation will take that into account. The
        input pipeline also pads the partial batches (last batch of eval/test
        set with num_example < batch_size). batch['batch_mask'] is used to
        identify padding examples which is incorporated to set the weight of
        these examples to zero in the loss computations.
      indices: Matcher output of shape [batch-size, 2, num-objects] which
        conveys source to target pairing of objects.
      log: If true, return classification accuracy as well.

    Returns:
      loss: Dict with 'loss_class' and other model specific losses.
      metrics: Dict with 'loss_class' and other model specific metrics.
    """
    assert 'logits' in outputs
    assert 'labels' in batch

    # batch_weights = batch.get('batch_mask')
    losses, metrics = {}, {}
    targets = batch['labels']
    
    src_logits = outputs['logits']

    # Apply the permutation communicated by indices.
    src_logits = simple_gather(src_logits, indices[:, 0])
    tgt_labels = simple_gather(targets, indices[:, 1])

    unnormalized_loss_class, denom = _compute_per_example_class_loss(
        tgt_labels=tgt_labels,
        src_logits=src_logits,
        # batch_weights=batch_weights,
    )

    metrics['loss_class'] = (unnormalized_loss_class.sum(), denom.sum())


    normalized_loss_class = unnormalized_loss_class.sum(dim=1)
    denom = torch.maximum(denom, torch.tensor([1.]).to(denom.device))
    normalized_loss_class = (normalized_loss_class / denom).mean()

    losses['loss_class'] = normalized_loss_class

    if log:
      # Class accuracy for non-padded (label != 0) labels
      not_padded = tgt_labels[:, :, 0] == 0
      num_correct_no_pad = weighted_correctly_classified(
          src_logits[..., 1:], tgt_labels[..., 1:], weights=not_padded)
      metrics['class_accuracy_not_pad'] = (num_correct_no_pad, not_padded.sum())


    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = psum_metric_normalizer(v)
    return losses, metrics

def boxes_losses_and_metrics(
    outputs,
    batch,
    indices):
    """Bounding box losses: L1 regression loss and GIoU loss.

    Args:
      outputs: dict; Model predictions. For the purpose of this loss, outputs
        must have key 'pred_boxes'. outputs['pred_boxes'] is a nd-array of the
        predicted box coordinates in (cx, cy, w, h) format. This nd-array has
        shape [batch-size, num-boxes, 4].
      batch: dict; that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, batch['label']
        must have key 'boxes', which the value has the same format as
        outputs['pred_boxes']. Additionally in batch['label'], key 'labels' is
        required that should match the specs defined in the member function
        `labels_losses_and_metrics`. This is to decide which boxes are invalid
        and need to be ignored. Invalid boxes have class label 0. If
        batch['batch_mask'] is provided it is used to weight the loss for
        different images in the current batch of examples.
      indices: list[tuple[nd-array, nd-array]]; Matcher output which conveys
        source to target pairing of objects.

    Returns:
      loss: dict with keys 'loss_bbox', 'loss_giou'. These are
        losses averaged over the batch. Therefore they have shape [].
      metrics: dict with keys 'loss_bbox' and 'loss_giou`.
        These are metrics psumed over the batch. Therefore they have shape [].
    """
    assert 'pred_boxes' in outputs
    assert 'boxes' in batch
    assert 'labels' in batch

    targets = {'labels': batch['labels'], 'boxes': batch['boxes']}
    losses, metrics = {}, {}

    src_boxes = simple_gather(outputs['pred_boxes'], indices[:, 0])
    tgt_boxes = simple_gather(targets['boxes'], indices[:, 1])
    tgt_labels = targets['labels']

    # Some of the boxes are padding. We want to discount them from the loss.
    n_labels_per_instance = torch.sum(tgt_labels[..., 1:], dim=-1)
    tgt_not_padding = n_labels_per_instance > 0  # [B, M] 

    # tgt_is_padding has shape [batch-size, num-boxes].
    # Align this with the model predictions using simple_gather.
    tgt_not_padding = simple_gather(tgt_not_padding, indices[:, 1]) ## raises Error when only one object is not padded

    src_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(src_boxes)
    tgt_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(tgt_boxes)
    unnormalized_loss_giou = 1 - box_utils.generalized_box_iou(
        src_boxes_xyxy, tgt_boxes_xyxy, all_pairs=False)

    unnormalized_loss_bbox = weighted_box_l1_loss(
        src_boxes_xyxy,
        tgt_boxes_xyxy,
    ).sum(dim=2)

    denom = tgt_not_padding.sum(dim=1)
    
    unnormalized_loss_bbox *= tgt_not_padding
    unnormalized_loss_giou *= tgt_not_padding

    # denom = jnp.maximum(jax.lax.pmean(denom.sum(), axis_name='batch'), 1)
    denom_sum = distributed_pmean(denom.sum())
    denom = torch.maximum(denom_sum, torch.tensor(1.0))

    normalized_loss_bbox = unnormalized_loss_bbox.sum() / denom
    normalized_loss_giou = unnormalized_loss_giou.sum() / denom

    losses['loss_bbox'] = normalized_loss_bbox
    metrics['loss_bbox'] = (normalized_loss_bbox, torch.tensor(1.).to(tgt_labels.device))
    losses['loss_giou'] = normalized_loss_giou
    metrics['loss_giou'] = (normalized_loss_giou, torch.tensor(1.).to(tgt_labels.device))

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = psum_metric_normalizer(v)  # pytype: disable=wrong-arg-types  # jax-ndarray
    return losses, metrics  # pytype: disable=bad-return-type  # jax-ndarray

def distributed_pmean(denom: torch.Tensor, axis_name: str = 'batch') -> torch.Tensor:
    """
    Computes the mean of `denom` across all processes in a distributed setup.

    Args:
        denom: The tensor to be reduced.
        axis_name: Unused, kept for API compatibility.

    Returns:
        The mean of `denom` across all processes.
    """
    if torch.distributed.is_initialized():
        # Sum the denom across all processes
        torch.distributed.all_reduce(denom, op=torch.distributed.ReduceOp.SUM)
        
        # Divide by the number of processes to get the mean
        denom = denom / torch.distributed.get_world_size()

    return denom

def psum_metric_normalizer(
    metrics,
    axis_name = 'batch'):
    """Applies psum over the given tuple of (metric, normalizer)."""
    
    # Summing over the batch
    summed_metric = torch.sum(metrics[0])
    summed_normalizer = torch.sum(metrics[1])
    
    if torch.distributed.is_initialized():
        # Applying psum across all processes if in distributed mode
        torch.distributed.all_reduce(summed_metric, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(summed_normalizer, op=torch.distributed.ReduceOp.SUM)
    
    return (summed_metric, summed_normalizer)

def weighted_correctly_classified(
    logits,
    one_hot_targets,
    weights = None):
  """Computes weighted number of correctly classified over the given batch.

  This computes the weighted number of correctly classified examples/pixels in a
  single, potentially padded minibatch. If the minibatch/inputs is padded (i.e.,
  it contains null examples/pad pixels) it is assumed that weights is a binary
  mask where 0 indicates that the example/pixel is null/padded. We assume the
  trainer will aggregate and divide by number of samples.

  Args:
   logits: Output of model in shape [batch, ..., num_classes].
   one_hot_targets: One hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

  Returns:
    The number of correctly classified examples in the given batch.
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))
  preds = torch.argmax(logits, dim=-1)
  targets = torch.argmax(one_hot_targets, dim=-1)
  correct = preds == targets

  if weights is not None:
    correct = apply_weights(correct, weights)

  return correct.to(torch.int32)


def weighted_box_l1_loss(
    pred,
    tgt,
    reduction = None,
    ):
    """L1 loss for bounding box with optional reduction specified.

    Args:
      pred: Prediction boxes of shape (..., 4), where the last dimension has form
        (x_min, y_min, x_max, y_max).
      tgt: Target boxes of shape (..., 4), where the last dimension has form
        (x_min, y_min, x_max, y_max).
      weights: Weights to apply to the loss.
      reduction: Type of reduction, which is from [None, 'mean'].
      tight: If True, returns the vanilla L1 loss on the bounding box coordinates.
        If False, returns loose bounding-box L1 loss, where prediction edges only
        generate loss when they stretch outside the target box, but not when they
        are within it.

    Returns:
      reduction(jnp.abs(src - tgt)). 'mean' reduction takes the global mean. To
      use customized normalization use 'none' reduction and scale loss in the
      caller.
    """
    if pred.shape[-1] != 4:
      raise ValueError(
          f'The last dimension of the prediction boxes must be 4.'
          f' Got shape {pred.shape}.'
      )
    if tgt.shape[-1] != 4:
      raise ValueError(
          f'The last dimension of the target boxes must be 4.'
          f' Got shape {tgt.shape}.'
      )
    
    abs_diff = torch.abs(pred - tgt)
    
    if not reduction:
      return abs_diff
    elif reduction == 'mean':
      return abs_diff.mean()
    else:
      raise ValueError(f'Unknown reduction: {reduction}')

def apply_weights(output: torch.Tensor, weights: torch.Tensor):
    """Applies given weights of the inputs in the minibatch to outputs."""
    
    if output.ndim < weights.ndim:
        raise ValueError("Output rank should be higher or equal to weights rank.")
    
    # Adjust the shape of weights to match the output shape
    desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
    weights = weights.view(desired_weights_shape)
    
    return output * weights

def _compute_per_example_class_loss(
      *,
      tgt_labels,
      src_logits,
      # batch_weights,
  ):
    """Computes the unnormalized per-example classification loss and denom."""
    loss_kwargs = {
        # 'weights': batch_weights,
    }
    
    loss_kwargs['gamma'] = 2.0
    loss_kwargs['alpha'] = 0.3

    # Don't compute loss for the padding index.
    unnormalized_loss_class = focal_sigmoid_cross_entropy(
        src_logits[..., 1:], tgt_labels[..., 1:])
    # Sum losses over all classes. The unnormalized_loss_class is of shape
    # [bs, 1 + max_num_boxes, num_classes], and after the next line, it becomes
    # [bs, 1 + max_num_boxes].
    unnormalized_loss_class = torch.sum(unnormalized_loss_class, dim=-1)
    # Normalize by number of "true" labels after removing padding label.
    denom = tgt_labels[..., 1:].sum(dim=[1, 2])  # pytype: disable=wrong-arg-types  # jax-ndarray

    return unnormalized_loss_class, denom


def simple_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gathers x using the indices in idx.

    Args:
        x: Inputs of shape [bs, n, d].
        idx: An array of shape [bs, m] and dtype torch.int32 or torch.int64 that specifies
            indexes we want to gather from x.

    Returns:
        Gathered output of shape [bs, m, d].
    """
    # Ensure idx is expanded to match the dimensionality of x
    if x.ndim < 3:
      idx_expanded = idx
    else:  
      idx_expanded = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
    
    # Use torch.gather to gather the values
    gathered_output = torch.gather(x, dim=1, index=idx_expanded)
    
    return gathered_output

def focal_sigmoid_cross_entropy(
    logits,
    multi_hot_targets,
    alpha = 0.5,
    gamma = 2.0):
  """Computes focal softmax cross-entropy given logits and targets.

  Focal loss as defined in https://arxiv.org/abs/1708.02002. Assuming y is the
  target vector and p is the predicted probability for the class, then:

  p_t = p if y == 1 and 1-p otherwise
  alpha_t = alpha if y == 1 and 1-alpha otherwise

  Focal loss = -alpha_t * (1-p_t)**gamma * log(p_t)

  NOTE: this is weighted unnormalized computation of loss that returns the loss
  of examples in the batch. If you are using it as a loss function, you can
  use the normalilzed version as:
  ```
    unnormalized_loss = focal_sigmoid_cross_entropy(...)
    if weights is not None:
      normalization = weights.sum()
    else:
      normalization = np.prod(multi_hot_targets.shape[:-1])
    loss = jnp.sum(unnormalized_loss) / (normalization + 1e-8)
  ```

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to be log probs.
    alpha: Balancing factor of the focal loss.
    gamma: Modulating factor of the focal loss.

  Returns:
    The loss of the examples in the given batch.
  """
  log_p, log_not_p = torch.nn.functional.logsigmoid(logits), torch.nn.functional.logsigmoid(-logits)

  loss = -(multi_hot_targets * log_p + (1. - multi_hot_targets) * log_not_p)

  p_t = torch.exp(-loss)
  loss *= (1 - p_t)**gamma
  loss *= alpha * multi_hot_targets + (1 - alpha) * (1 - multi_hot_targets)

  return loss


if __name__ == "__main__":

  pass