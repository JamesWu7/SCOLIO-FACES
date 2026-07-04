from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from .config import (
    BINARY_CLASSES,
    BINARY_DISEASE_INDEX,
    BINARY_WEIGHTS,
    DLIB_LANDMARK_MODEL,
    MODEL_VERSIONS,
    SS_SUBTYPE_CLASSES,
    SUBTYPE_WEIGHTS,
    SUPERCLASS_CLASSES,
    SUPERCLASS_SYNDROMIC_INDEX,
    SUPERCLASS_WEIGHTS,
)
from .models import BinaryCNNCBAM, ResNet18Etiology, ResNet50SSSubtype
from .preprocess import BINARY_TRANSFORM, RESNET_TRANSFORM, FaceCropper
from .weights import resolve_weight_path


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _ranked_results(class_names, probabilities):
    items = [
        {
            "class_name": str(class_name),
            "probability": float(probability),
            "probability_percent": float(probability) * 100.0,
        }
        for class_name, probability in zip(class_names, probabilities)
    ]
    return sorted(items, key=lambda item: item["probability"], reverse=True)


class FACESPredictor:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cropper = FaceCropper(resolve_weight_path(DLIB_LANDMARK_MODEL))
        self.binary_model = self._load_binary_model()
        self.etiology_model = None
        self.subtype_model = None

    def _load_binary_model(self):
        binary_weights = resolve_weight_path(BINARY_WEIGHTS)
        model = BinaryCNNCBAM(input_channels=3, dropout_rate=0.35)
        model.load_state_dict(_torch_load(binary_weights, self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_etiology_model(self):
        if self.etiology_model is not None:
            return self.etiology_model
        superclass_weights = resolve_weight_path(SUPERCLASS_WEIGHTS)
        model = ResNet18Etiology(dropout_rate=0.5)
        model.load_state_dict(_torch_load(superclass_weights, self.device))
        model.to(self.device)
        model.eval()
        self.etiology_model = model
        return model

    def _load_subtype_model(self):
        if self.subtype_model is not None:
            return self.subtype_model
        subtype_weights = resolve_weight_path(SUBTYPE_WEIGHTS)
        model = ResNet50SSSubtype(num_classes=len(SS_SUBTYPE_CLASSES), dropout_rate=0.5)
        model.load_state_dict(_torch_load(subtype_weights, self.device))
        model.to(self.device)
        model.eval()
        self.subtype_model = model
        return model

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        if image is None:
            raise ValueError("Please upload one frontal facial photograph first.")

        warnings = []
        crop_result = self.cropper.crop(image)
        if crop_result.warning:
            warnings.append(crop_result.warning)
        face_image = crop_result.image.convert("RGB")

        binary_tensor = BINARY_TRANSFORM(face_image).unsqueeze(0).to(self.device)

        disease_logit = self.binary_model(binary_tensor)
        disease_prob = torch.sigmoid(disease_logit).flatten()[0].item()
        binary_probs = [1.0 - disease_prob, disease_prob]
        binary = _ranked_results(BINARY_CLASSES, binary_probs)

        disease_is_top = binary[0]["class_name"] == BINARY_CLASSES[BINARY_DISEASE_INDEX]
        if not disease_is_top:
            warnings.append(
                "The binary screening result did not suggest a scoliosis-related facial phenotype; "
                "the etiology superclass model and the syndromic scoliosis subtype model were not run."
            )
            return {
                "cropped_face": face_image,
                "face_crop_method": crop_result.method,
                "binary": binary,
                "etiology": [],
                "ss_subtypes": [],
                "show_ss_subtypes": False,
                "downstream_skipped": True,
                "warnings": warnings,
                "model_versions": MODEL_VERSIONS,
                "device": str(self.device),
                "top_summary": {
                    "screening_result": binary[0]["class_name"],
                    "screening_probability": binary[0]["probability"],
                    "etiology_top1": "Not run",
                    "etiology_probability": None,
                    "ss_subtype_top3": [],
                },
            }

        resnet_tensor = RESNET_TRANSFORM(face_image).unsqueeze(0).to(self.device)
        etiology_model = self._load_etiology_model()

        etiology_logits = etiology_model(resnet_tensor)
        etiology_probs = torch.softmax(etiology_logits, dim=1).flatten().cpu().tolist()

        etiology = _ranked_results(SUPERCLASS_CLASSES, etiology_probs)

        etiology_top1 = etiology[0]["class_name"]
        syndromic_class = SUPERCLASS_CLASSES[SUPERCLASS_SYNDROMIC_INDEX]
        show_ss_subtypes = etiology_top1 == syndromic_class
        if show_ss_subtypes:
            subtype_model = self._load_subtype_model()
            subtype_logits = subtype_model(resnet_tensor)
            subtype_probs = torch.softmax(subtype_logits, dim=1).flatten().cpu().tolist()
            ss_subtypes = _ranked_results(SS_SUBTYPE_CLASSES, subtype_probs)
        else:
            ss_subtypes = []
            warnings.append(
                "Syndromic scoliosis is not the etiology superclass with the highest support; "
                "the eleven-class syndromic scoliosis subtype model was not run."
            )

        return {
            "cropped_face": face_image,
            "face_crop_method": crop_result.method,
            "binary": binary,
            "etiology": etiology,
            "ss_subtypes": ss_subtypes,
            "show_ss_subtypes": show_ss_subtypes,
            "downstream_skipped": False,
            "warnings": warnings,
            "model_versions": MODEL_VERSIONS,
            "device": str(self.device),
            "top_summary": {
                "screening_result": binary[0]["class_name"],
                "screening_probability": binary[0]["probability"],
                "etiology_top1": etiology_top1,
                "etiology_probability": etiology[0]["probability"],
                "ss_subtype_top3": ss_subtypes[:3],
            },
        }


@lru_cache(maxsize=1)
def get_predictor() -> FACESPredictor:
    return FACESPredictor()


def predict_faces(image: Image.Image) -> dict:
    return get_predictor().predict(image)
