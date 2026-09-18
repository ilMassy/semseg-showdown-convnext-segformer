"""
Dataset e pipeline di augmentation per PASCAL VOC 2012 (Segmentation).

Wrappa torchvision.datasets.VOCSegmentation aggiungendo:
- augmentation con Albumentations (compatibile sia con l'encoder CNN che SegFormer)
- conversione delle maschere in formato compatibile con CrossEntropy/Dice loss
- gestione dell'indice "void" (bordi ignorati, valore 255 in VOC) escluso dalla loss

Uso:
    from dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(root="./data", batch_size=8, img_size=512)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Indice usato da PASCAL VOC per i pixel di bordo/ignorati: va escluso dalla loss
# (in CrossEntropyLoss: ignore_index=VOID_INDEX)
VOID_INDEX = 255

# 20 classi oggetto + background (indice 0) = 21 classi totali in PASCAL VOC 2012
NUM_CLASSES = 21

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Statistiche ImageNet: usate perché sia ConvNeXt che SegFormer sono pretrained su ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(img_size: int = 512, use_augmentation: bool = True) -> A.Compose:
    """Augmentation per il training. Applica le stesse trasformazioni geometriche
    a immagine e maschera contemporaneamente (fondamentale per la segmentazione).

    use_augmentation=False disattiva tutte le trasformazioni stocastiche (mantenendo
    solo resize + normalizzazione, come in validazione) — necessario per l'ablation
    study "con/senza augmentation" previsto nella proposta di progetto.
    """
    if not use_augmentation:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 512) -> A.Compose:
    """Nessuna augmentation stocastica in validazione: solo resize e normalizzazione,
    per una valutazione confrontabile tra epoche e tra modelli."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class VOCSegmentationAlbu(Dataset):
    """Wrapper attorno a VOCSegmentation che applica Albumentations invece delle
    transform separate di torchvision (necessario per trasformare immagine+maschera
    in modo sincronizzato con augmentation più ricche di RandomResizedCrop/flip)."""

    def __init__(self, root: str, image_set: str, transform: A.Compose, download: bool = True):
        self.dataset = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=download,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, mask = self.dataset[idx]
        image = np.array(image)
        mask = np.array(mask)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()

        return image, mask


def get_dataloaders(
    root: str = "./data",
    batch_size: int = 8,
    img_size: int = 512,
    num_workers: int = 4,
    download: bool = True,
    use_augmentation: bool = True,
):
    """Crea train e validation DataLoader per PASCAL VOC 2012.

    Nota su batch_size/img_size: con 8GB di VRAM, batch_size=8 e img_size=512
    è un buon punto di partenza per SegFormer in mixed precision. In caso di
    out-of-memory, riduci prima img_size (es. 384) prima di ridurre il batch_size,
    oppure usa gradient accumulation per mantenere un batch effettivo più grande.

    use_augmentation=False disattiva l'augmentation stocastica sul train set
    (per l'ablation study augmentation on/off). Il val set non è mai aumentato.
    """
    train_dataset = VOCSegmentationAlbu(
        root=root,
        image_set="train",
        transform=get_train_transform(img_size, use_augmentation=use_augmentation),
        download=download,
    )
    val_dataset = VOCSegmentationAlbu(
        root=root,
        image_set="val",
        transform=get_val_transform(img_size),
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Smoke test: verifica che il dataset si carichi e che shape/tipi siano corretti
    train_loader, val_loader = get_dataloaders(batch_size=2, img_size=256, download=True)

    images, masks = next(iter(train_loader))
    print(f"Batch immagini: {images.shape}, dtype={images.dtype}")
    print(f"Batch maschere: {masks.shape}, dtype={masks.dtype}")
    print(f"Valori unici nella maschera: {torch.unique(masks).tolist()}")
    print(f"Numero classi (incluso background): {NUM_CLASSES}")
    print(f"Train set: {len(train_loader.dataset)} immagini")
    print(f"Val set: {len(val_loader.dataset)} immagini")
