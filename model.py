import torch
from torch import nn
from transformers import Owlv2ForObjectDetection
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from transformers.utils import ModelOutput


@dataclass
class Owlv2ObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        objectness_logits (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`):
            The objectness logits of all image patches. OWL-ViT represents images as a set of image patches where the
            total number of patches is (image_size / patch_size)**2.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`Owlv2TextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes image
            embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    objectness_logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output = None
    vision_model_output = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class XViewDetector(nn.Module):

    def __init__(self, name="google/owlv2-base-patch16-ensemble"):
        super().__init__()
        self.detector = Owlv2ForObjectDetection.from_pretrained(name)

        print("All weights frozen")
        for param in self.detector.parameters():
            param.requires_grad = False
        
        print("Class predictor unfrozen")
        for param in self.detector.class_head.parameters():
            param.requires_grad = True
        
        print("Objectness predictor unfrozen")
        for param in self.detector.objectness_head.parameters():
            param.requires_grad = True
        
        print("Box predictor unfrozen")
        for param in self.detector.box_head.parameters():
            param.requires_grad = True
        
    def objectness_predictor(self, image_features):
        # image_features = image_features.detach()
        objectness_logits = self.detector.objectness_head(image_features)
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.detector.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.detector.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.detector.config.return_dict

        # Embed images and text queries
        query_embeds, feature_map, outputs = self.detector.image_text_embedder(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Text and vision model outputs
        text_outputs = outputs.text_model_output
        vision_outputs = outputs.vision_model_output

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.detector.class_predictor(image_feats, query_embeds, query_mask)

        # Predict objectness
        # objectness_logits = self.detector.objectness_predictor(image_feats)
        objectness_logits = self.objectness_predictor(image_feats)

        # Predict object boxes
        pred_boxes = self.detector.box_predictor(image_feats, feature_map)

        # output = (
        #     pred_logits,
        #     objectness_logits,
        #     pred_boxes,
        #     query_embeds,
        #     feature_map,
        #     class_embeds,
        #     text_outputs.to_tuple(),
        #     vision_outputs.to_tuple(),
        # )
        # output = tuple(x for x in output if x is not None)
        # return output
        return Owlv2ObjectDetectionOutput(
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            objectness_logits=objectness_logits,
            class_embeds=class_embeds,
            # text_model_output=text_outputs,
            # vision_model_output=vision_outputs,
        )


