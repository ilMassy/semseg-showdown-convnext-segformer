"""
Training loop per il confronto ConvNeXt-UNet vs SegFormer su PASCAL VOC 2012.

Caratteristiche:
- Mixed precision (bf16/fp16) — essenziale con 8GB di VRAM
- Gradient accumulation — per simulare batch size più grandi senza OOM
- Loss combinata Cross-Entropy + Dice — gestisce meglio lo sbilanciamento tra classi
  rispetto alla sola Cross-Entropy (previsto nell'ablation study)
- Logging su Weights & Biases — curve di training/validazione, confronto tra run
- Checkpointing del modello migliore (in base a mIoU di validazione)

Uso:
    python src/train.py --model convnext_unet --epochs 50 --batch_size 8
    python src/train.py --model segformer --epochs 50 --batch_size 8
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm

from dataset import get_dataloaders, NUM_CLASSES, VOID_INDEX
from models import build_model, count_parameters

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DiceLoss(nn.Module):
    """Dice loss multi-classe. Combinata con Cross-Entropy gestisce meglio le
    classi minoritarie rispetto alla sola CE, particolarmente utile in VOC dove
    il background domina la maggior parte dei pixel."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = VOID_INDEX, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = (target != self.ignore_index)
        target_clamped = target.clone()
        target_clamped[~valid_mask] = 0  # placeholder, verrà escluso dalla loss

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        intersection = (probs * target_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss multi-classe (Lin et al., 2017). Rispetto alla Cross-Entropy,
    riduce il peso dei pixel già ben classificati (es. background, la maggioranza
    in VOC) e concentra il gradiente sui pixel difficili/minoritari — complementare
    alla Dice loss, con cui viene combinata nell'ablation study "Dice+Focal"."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = VOID_INDEX,
                 gamma: float = 2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        # reduction="none" per poter mascherare i pixel void prima di fare la media
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)  # [B, H, W], già 0 sui pixel void (ignore_index)
        pt = torch.exp(-ce_loss)  # probabilità stimata della classe corretta
        focal = ((1 - pt) ** self.gamma) * ce_loss

        valid_mask = (target != self.ignore_index)
        return focal[valid_mask].mean()


class DiceFocalLoss(nn.Module):
    """Dice + Focal, pesate. Alternativa a Cross-Entropy nell'ablation study
    sulla scelta della loss, come da proposta di progetto."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = VOID_INDEX,
                 dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal = FocalLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.focal_weight * self.focal(logits, target)


def build_criterion(loss_type: str, num_classes: int = NUM_CLASSES, ignore_index: int = VOID_INDEX) -> nn.Module:
    """Factory per la loss, usata per l'ablation study richiesto dalla proposta
    di progetto: Cross-Entropy vs combinazione Dice+Focal.

    loss_type: "ce" oppure "dice_focal"
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "dice_focal":
        return DiceFocalLoss(num_classes=num_classes, ignore_index=ignore_index)
    else:
        raise ValueError(f"loss_type '{loss_type}' non riconosciuto. Scegli tra: 'ce', 'dice_focal'")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps, epoch, use_amp):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    epoch_start = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=total_loss / (step + 1))

    # Tempo totale e per-immagine dell'epoca di training — dato richiesto
    # dall'analisi di efficienza della proposta di progetto.
    epoch_time = time.time() - epoch_start
    sec_per_image = epoch_time / len(loader.dataset)

    return total_loss / len(loader), epoch_time, sec_per_image


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=VOID_INDEX).to(device)
    # F1 macro come proxy del Dice score medio (Dice == F1 per classe in una
    # formulazione one-vs-rest), richiesto dalla valutazione "rigorosa" della proposta
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro", ignore_index=VOID_INDEX).to(device)

    n_images = 0
    inference_start = time.time()
    for images, masks in tqdm(loader, desc="[val]"):
        images, masks = images.to(device), masks.to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        jaccard.update(preds, masks)
        f1_metric.update(preds, masks)
        n_images += images.size(0)

    # Tempo di inferenza per immagine — dato richiesto dall'analisi di efficienza
    inference_time = time.time() - inference_start
    sec_per_image_inference = inference_time / n_images

    miou = jaccard.compute().item()
    dice_macro = f1_metric.compute().item()
    return total_loss / len(loader), miou, dice_macro, sec_per_image_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["convnext_unet", "segformer"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum_steps", type=int, default=2,
                         help="Gradient accumulation: batch effettivo = batch_size * accum_steps")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--loss_type", choices=["ce", "dice_focal"], default="dice_focal",
                         help="Ablation study loss: 'ce' (sola Cross-Entropy) vs 'dice_focal' (Dice+Focal)")
    parser.add_argument("--no_augmentation", action="store_true",
                         help="Disattiva l'augmentation stocastica sul train set (ablation study augmentation on/off)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(project="sii-segmentation-voc2012", name=args.model, config=vars(args))

    print(f"Device: {device} | Mixed precision: {use_amp} | Modello: {args.model}")

    train_loader, val_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size, img_size=args.img_size,
        use_augmentation=not args.no_augmentation,
    )

    model = build_model(args.model, num_classes=NUM_CLASSES).to(device)
    params_info = count_parameters(model)
    print(f"Parametri: {params_info['total_params_millions']}M")
    print(f"Loss: {args.loss_type} | Augmentation: {not args.no_augmentation}")

    criterion = build_criterion(args.loss_type, num_classes=NUM_CLASSES, ignore_index=VOID_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_miou = 0.0
    total_train_time = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, epoch_time, train_sec_per_image = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            args.accum_steps, epoch, use_amp,
        )
        val_loss, val_miou, val_dice, val_sec_per_image = validate(
            model, val_loader, criterion, device, use_amp,
        )
        scheduler.step()
        total_train_time += epoch_time

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f} val_Dice={val_dice:.4f} | "
            f"train_time={epoch_time:.1f}s ({train_sec_per_image*1000:.1f}ms/img) "
            f"inference={val_sec_per_image*1000:.1f}ms/img"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "val_Dice": val_dice,
                "lr": scheduler.get_last_lr()[0],
                "train_epoch_time_sec": epoch_time,
                "train_ms_per_image": train_sec_per_image * 1000,
                "inference_ms_per_image": val_sec_per_image * 1000,
            })

        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_miou": val_miou,
                "val_dice": val_dice,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> Nuovo miglior modello salvato: {ckpt_path} (mIoU={val_miou:.4f})")

    # Riepilogo per l'analisi di efficienza: parametri + tempo totale + medio per epoca
    print(f"\nTraining completato. Miglior mIoU: {best_miou:.4f}")
    print(
        f"Efficienza — Parametri: {params_info['total_params_millions']}M | "
        f"Tempo training totale: {total_train_time/60:.1f}min | "
        f"Tempo medio/epoca: {total_train_time/args.epochs:.1f}s"
    )
    if use_wandb:
        wandb.log({
            "total_train_time_min": total_train_time / 60,
            "avg_epoch_time_sec": total_train_time / args.epochs,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
