"""
Definizione dei due modelli da confrontare nel progetto:

1. ConvNeXt-UNet  — architettura encoder-decoder convoluzionale moderna
                     (encoder ConvNeXt pretrained + decoder UNet)
2. SegFormer      — architettura transformer-based nativa per la segmentazione

Entrambi espongono la stessa interfaccia (forward -> logits [B, NUM_CLASSES, H, W])
così da poter essere usati in modo intercambiabile nel training loop e nella
valutazione, mantenendo il confronto equo.

Uso:
    from models import build_model

    model = build_model("convnext_unet", num_classes=21)
    model = build_model("segformer", num_classes=21)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

from dataset import NUM_CLASSES


class ConvNeXtUNet(nn.Module):
    """Encoder-decoder convoluzionale moderno: encoder ConvNeXt (pretrained ImageNet)
    + decoder UNet. Scelto al posto di un ResNet "classico" perché ConvNeXt è
    l'architettura convoluzionale più aggiornata con performance allo stato
    dell'arte tra le CNN pure, mantenendo un confronto equo con SegFormer.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, encoder_name: str = "tu-convnext_tiny"):
        super().__init__()
        # encoder_name con prefisso "tu-" usa timm come backend (necessario per ConvNeXt)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


class SegFormerWrapper(nn.Module):
    """Wrapper attorno a SegFormer (HuggingFace) per uniformare l'interfaccia
    con ConvNeXtUNet: HuggingFace restituisce un output object e logits a
    risoluzione ridotta (1/4), qui li riportiamo alla risoluzione di input
    tramite upsampling, così le due architetture sono confrontabili pixel-per-pixel.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # necessario: ADE20K ha 150 classi, VOC ne ha 21
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # shape: [B, num_classes, H/4, W/4]
        # Upsampling alla risoluzione originale per confronto pixel-a-pixel con l'altro modello
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


MODEL_REGISTRY = {
    "convnext_unet": ConvNeXtUNet,
    "segformer": SegFormerWrapper,
}


def build_model(name: str, num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    """Factory function per istanziare uno dei due modelli per nome.

    Args:
        name: "convnext_unet" oppure "segformer"
        num_classes: numero di classi di output (default: 21, PASCAL VOC)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modello '{name}' non riconosciuto. Scegli tra: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> dict:
    """Conta i parametri totali e allenabili — utile per l'analisi di efficienza
    richiesta nel progetto (confronto ConvNeXt-UNet vs SegFormer in termini di
    numero di parametri, non solo di performance)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_millions": round(total / 1e6, 2),
    }


if __name__ == "__main__":
    # Smoke test: verifica che entrambi i modelli producano output della shape attesa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 3, 512, 512, device=device)

    for model_name in MODEL_REGISTRY:
        print(f"\n--- {model_name} ---")
        model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"Output shape: {output.shape}")  # atteso: [2, 21, 512, 512]
        assert output.shape == (2, NUM_CLASSES, 512, 512), "Shape output inattesa!"

        params = count_parameters(model)
        print(f"Parametri totali: {params['total_params_millions']}M")
