import torch


# def box_cxcywh_to_xyxy(x: Array, np_backbone: PyModule = jnp) -> Array:
#     """Converts boxes from [cx, cy, w, h] format into [x, y, x', y'] format."""
#     x_c, y_c, w, h = np_backbone.split(x, 4, axis=-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return np_backbone.concatenate(b, axis=-1)

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts boxes from [cx, cy, w, h] format into [x, y, x', y'] format."""
    x_c, y_c, w, h = x.split(1, dim=-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.cat(b, dim=-1)


# def box_iou(boxes1: Array,
#             boxes2: Array,
#             np_backbone: PyModule = jnp,
#             all_pairs: bool = True,
#             eps: float = 1e-6) -> Array:
#     """Computes IoU between two sets of boxes.

#     Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom
#     right.

#     Args:
#     boxes1: Predicted bounding-boxes in shape [bs, n, 4].
#     boxes2: Target bounding-boxes in shape [bs, m, 4]. Can have a different
#         number of boxes if all_pairs is True.
#     np_backbone: numpy module: Either the regular numpy package or jax.numpy.
#     all_pairs: Whether to compute IoU between all pairs of boxes or not.
#     eps: Epsilon for numerical stability.

#     Returns:
#     If all_pairs == True, returns the pairwise IoU cost matrix of shape
#     [bs, n, m]. If all_pairs == False, returns the IoU between corresponding
#     boxes. The shape of the return value is then [bs, n].
#     """

#     # First, compute box areas. These will be used later for computing the union.
#     wh1 = boxes1[..., 2:] - boxes1[..., :2]
#     area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

#     wh2 = boxes2[..., 2:] - boxes2[..., :2]
#     area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

#     if all_pairs:
#     # Compute pairwise top-left and bottom-right corners of the intersection
#     # of the boxes.
#     lt = np_backbone.maximum(boxes1[..., :, None, :2],
#                                 boxes2[..., None, :, :2])  # [bs, n, m, 2].
#     rb = np_backbone.minimum(boxes1[..., :, None, 2:],
#                                 boxes2[..., None, :, 2:])  # [bs, n, m, 2].

#     # intersection = area of the box defined by [lt, rb]
#     wh = (rb - lt).clip(0.0)  # [bs, n, m, 2]
#     intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]

#     # union = sum of areas - intersection
#     union = area1[..., :, None] + area2[..., None, :] - intersection

#     iou = intersection / (union + eps)

#     else:
#     # Compute top-left and bottom-right corners of the intersection between
#     # corresponding boxes.
#     assert boxes1.shape[1] == boxes2.shape[1], (
#         'Different number of boxes when all_pairs is False')
#     lt = np_backbone.maximum(boxes1[..., :, :2],
#                                 boxes2[..., :, :2])  # [bs, n, 2]
#     rb = np_backbone.minimum(boxes1[..., :, 2:], boxes2[..., :,
#                                                         2:])  # [bs, n, 2]

#     # intersection = area of the box defined by [lt, rb]
#     wh = (rb - lt).clip(0.0)  # [bs, n, 2]
#     intersection = wh[..., :, 0] * wh[..., :, 1]  # [bs, n]

#     # union = sum of areas - intersection.
#     union = area1 + area2 - intersection

#     # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
#     iou = intersection / (union + eps)

#     return iou, union  # pytype: disable=bad-return-type  # jax-ndarray

def box_iou(boxes1: torch.Tensor,
            boxes2: torch.Tensor,
            all_pairs: bool = True,
            eps: float = 1e-6) -> torch.Tensor:
    """Computes IoU between two sets of boxes.

    Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom
    right.

    Args:
    boxes1: Predicted bounding-boxes in shape [bs, n, 4].
    boxes2: Target bounding-boxes in shape [bs, m, 4]. Can have a different
        number of boxes if all_pairs is True.
    all_pairs: Whether to compute IoU between all pairs of boxes or not.
    eps: Epsilon for numerical stability.

    Returns:
    If all_pairs == True, returns the pairwise IoU cost matrix of shape
    [bs, n, m]. If all_pairs == False, returns the IoU between corresponding
    boxes. The shape of the return value is then [bs, n].
    """
    wh1 = boxes1[..., 2:] - boxes1[..., :2]
    area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

    wh2 = boxes2[..., 2:] - boxes2[..., :2]
    area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

    if all_pairs:
        lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [bs, n, m, 2]
        rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [bs, n, m, 2]
        wh = (rb - lt).clamp(min=0)  # [bs, n, m, 2]
        intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]
        union = area1[..., :, None] + area2[..., None, :] - intersection
        iou = intersection / (union + eps)
    else:
        assert boxes1.shape[1] == boxes2.shape[1], (
            'Different number of boxes when all_pairs is False')
        lt = torch.maximum(boxes1[..., :, :2], boxes2[..., :, :2])  # [bs, n, 2]
        rb = torch.minimum(boxes1[..., :, 2:], boxes2[..., :, 2:])  # [bs, n, 2]
        wh = (rb - lt).clamp(min=0)  # [bs, n, 2]
        intersection = wh[..., 0] * wh[..., 1]  # [bs, n]
        union = area1 + area2 - intersection
        iou = intersection / (union + eps)

    return iou, union


# def generalized_box_iou(boxes1: Array,
#                         boxes2: Array,
#                         np_backbone: PyModule = jnp,
#                         all_pairs: bool = True,
#                         eps: float = 1e-6) -> Array:
#     """Generalized IoU from https://giou.stanford.edu/.

#     The boxes should be in [x, y, x', y'] format specifying top-left and
#     bottom-right corners.

#     Args:
#     boxes1: Predicted bounding-boxes in shape [..., n, 4].
#     boxes2: Target bounding-boxes in shape [..., m, 4].
#     np_backbone: Numpy module: Either the regular numpy package or jax.numpy.
#     all_pairs: Whether to compute generalized IoU from between all-pairs of
#         boxes or not. Note that if all_pairs == False, we must have m==n.
#     eps: Epsilon for numerical stability.

#     Returns:
#     If all_pairs == True, returns a [bs, n, m] pairwise matrix, of generalized
#     ious. If all_pairs == False, returns a [bs, n] matrix of generalized ious.
#     """
#     # Degenerate boxes gives inf / nan results, so do an early check.
#     # TODO(b/166344282): Figure out how to enable asserts on inputs with jitting:
#     # assert (boxes1[:, :, 2:] >= boxes1[:, :, :2]).all()
#     # assert (boxes2[:, :, 2:] >= boxes2[:, :, :2]).all()
#     iou, union = box_iou(
#         boxes1, boxes2, np_backbone=np_backbone, all_pairs=all_pairs, eps=eps)

#     # Generalized IoU has an extra term which takes into account the area of
#     # the box containing both of these boxes. The following code is very similar
#     # to that for computing intersection but the min and max are flipped.
#     if all_pairs:
#     lt = np_backbone.minimum(boxes1[..., :, None, :2],
#                                 boxes2[..., None, :, :2])  # [bs, n, m, 2]
#     rb = np_backbone.maximum(boxes1[..., :, None, 2:],
#                                 boxes2[..., None, :, 2:])  # [bs, n, m, 2]

#     else:
#     lt = np_backbone.minimum(boxes1[..., :, :2],
#                                 boxes2[..., :, :2])  # [bs, n, 2]
#     rb = np_backbone.maximum(boxes1[..., :, 2:], boxes2[..., :,
#                                                         2:])  # [bs, n, 2]

#     # Now, compute the covering box's area.
#     wh = (rb - lt).clip(0.0)  # Either [bs, n, 2] or [bs, n, m, 2].
#     area = wh[..., 0] * wh[..., 1]  # Either [bs, n] or [bs, n, m].

#     # Finally, compute generalized IoU from IoU, union, and area.
#     # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
#     return iou - (area - union) / (area + eps)

def generalized_box_iou(boxes1: torch.Tensor,
                        boxes2: torch.Tensor,
                        all_pairs: bool = True,
                        eps: float = 1e-6) -> torch.Tensor:
    """Generalized IoU from https://giou.stanford.edu/.

    The boxes should be in [x, y, x', y'] format specifying top-left and
    bottom-right corners.

    Args:
    boxes1: Predicted bounding-boxes in shape [..., n, 4].
    boxes2: Target bounding-boxes in shape [..., m, 4].
    all_pairs: Whether to compute generalized IoU from between all-pairs of
        boxes or not. Note that if all_pairs == False, we must have m==n.
    eps: Epsilon for numerical stability.

    Returns:
    If all_pairs == True, returns a [bs, n, m] pairwise matrix, of generalized
    ious. If all_pairs == False, returns a [bs, n] matrix of generalized ious.
    """
    iou, union = box_iou(boxes1, boxes2, all_pairs=all_pairs, eps=eps)

    if all_pairs:
        lt = torch.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [bs, n, m, 2]
        rb = torch.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [bs, n, m, 2]
    else:
        lt = torch.minimum(boxes1[..., :, :2], boxes2[..., :, :2])  # [bs, n, 2]
        rb = torch.maximum(boxes1[..., :, 2:], boxes2[..., :, 2:])  # [bs, n, 2]

    wh = (rb - lt).clamp(min=0)  # [bs, n, 2] or [bs, n, m, 2]
    area = wh[..., 0] * wh[..., 1]  # [bs, n] or [bs, n, m]

    return iou - (area - union) / (area + eps)