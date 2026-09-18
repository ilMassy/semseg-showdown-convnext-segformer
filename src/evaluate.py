"""
Valutazione dettagliata di un modello allenato: metriche per classe (mIoU, Dice/F1),
matrice di confusione, e analisi di efficienza (tempi di inferenza, numero di parametri).

Questo è il file che genera le tabelle/grafici da mettere nel report finale per
il confronto ConvNeXt-UNet vs SegFormer (punti 2 e 5 della proposta al professore).

Nota metodologica: nella formulazione one-vs-rest per-classe, Dice e F1 sono
matematicamente identici (2*TP / (2*TP + FP + FN)), quindi si calcola una sola volta
(F1Score) e si riporta come "Dice/F1" nel report — evita di calcolare due metriche
ridondanti con due implementazioni torchmetrics diverse.

I risultati completi (metriche aggregate + per classe + efficienza) vengono salvati anche
in un file JSON (eval_results_<model>.json), per avere un riferimento univoco e
riproducibile da citare nel report finale (evita ambiguità come quella già capitata con i
checkpoint di SegFormer, vedi PROGRESS_REPORT.md).

Uso:
    python src/evaluate.py --model convnext_unet --checkpoint checkpoints/convnext_unet_best.pt
    python src/evaluate.py --model segformer --checkpoint checkpoints/segformer_best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics import JaccardIndex, F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm

from dataset import get_dataloaders, NUM_CLASSES, VOID_INDEX, VOC_CLASSES
from models import build_model, count_parameters


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes=NUM_CLASSES, ignore_index=VOID_INDEX):
    """Calcola metriche aggregate e per classe su tutto il validation set."""
    model.eval()

    iou_metric = JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average=None,
    ).to(device)
    # Dice e F1 sono matematicamente identici nella formulazione one-vs-rest per-classe
    # (2*TP / (2*TP + FP + FN)): si calcola solo F1Score e si riporta come "Dice/F1".
    f1_metric = F1Score(
        task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average=None,
    ).to(device)
    confmat_metric = MulticlassConfusionMatrix(
        num_classes=num_classes, ignore_index=ignore_index,
    ).to(device)

    for images, masks in tqdm(loader, desc="Valutazione"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        iou_metric.update(preds, masks)
        f1_metric.update(preds, masks)
        confmat_metric.update(preds, masks)

    per_class_iou = iou_metric.compute().cpu().numpy()
    per_class_dice_f1 = f1_metric.compute().cpu().numpy()
    confmat = confmat_metric.compute().cpu().numpy()

    return {
        "per_class_iou": per_class_iou,
        "per_class_dice_f1": per_class_dice_f1,
        "mean_iou": float(np.mean(per_class_iou)),
        "mean_dice_f1": float(np.mean(per_class_dice_f1)),
        "confusion_matrix": confmat,
    }


@torch.no_grad()
def measure_efficiency(model, device, img_size=512, num_warmup=10, num_runs=50):
    """Misura tempo medio di inferenza (batch_size=1) e numero di parametri.
    Fondamentale per il punto 5 della proposta: confronto di efficienza tra
    le due architetture, non solo di accuratezza."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # Warmup: le prime chiamate CUDA hanno overhead di inizializzazione
    for _ in range(num_warmup):
        _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / num_runs) * 1000
    params_info = count_parameters(model)

    return {
        "avg_inference_time_ms": round(avg_time_ms, 2),
        "fps": round(1000 / avg_time_ms, 2),
        **params_info,
    }


def save_results(results: dict, efficiency: dict, model_name: str, checkpoint_path: str,
                  checkpoint_epoch: int, output_dir: str = "."):
    """Salva metriche aggregate, per classe ed efficienza in un unico JSON riproducibile.

    Fondamentale per avere un riferimento univoco da citare nel report finale ed evitare
    ambiguità come quella già capitata con i checkpoint di SegFormer (vedi PROGRESS_REPORT.md,
    §3.3) — il numero giusto è quello nel file, non quello ricopiato a mano dal terminale."""
    out_path = Path(output_dir) / f"eval_results_{model_name}.json"

    payload = {
        "model": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "mean_iou": results["mean_iou"],
        "mean_dice_f1": results["mean_dice_f1"],
        "per_class": {
            cls_name: {
                "iou": float(results["per_class_iou"][i]),
                "dice_f1": float(results["per_class_dice_f1"][i]),
            }
            for i, cls_name in enumerate(VOC_CLASSES)
        },
        "efficiency": efficiency,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


def print_report(results: dict, efficiency: dict, model_name: str):
    print("\n" + "=" * 70)
    print(f"REPORT DI VALUTAZIONE — {model_name}")
    print("=" * 70)

    print(f"\nmIoU medio:     {results['mean_iou']:.4f}")
    print(f"Dice/F1 medio:  {results['mean_dice_f1']:.4f}")

    print("\nMetriche per classe:")
    print(f"{'Classe':<15} {'IoU':>8} {'Dice/F1':>10}")
    print("-" * 35)
    for i, cls_name in enumerate(VOC_CLASSES):
        print(f"{cls_name:<15} {results['per_class_iou'][i]:>8.4f} "
              f"{results['per_class_dice_f1'][i]:>10.4f}")

    print("\nEfficienza:")
    print(f"  Parametri totali:       {efficiency['total_params_millions']}M")
    print(f"  Tempo inferenza medio:  {efficiency['avg_inference_time_ms']} ms")
    print(f"  FPS (batch_size=1):     {efficiency['fps']}")

    # Classi con IoU più basso: utile per l'analisi degli errori (punto 3 della proposta)
    worst_indices = np.argsort(results["per_class_iou"])[:3]
    print("\nClassi più problematiche (IoU più basso) — utile per l'analisi degli errori:")
    for idx in worst_indices:
        print(f"  {VOC_CLASSES[idx]}: IoU={results['per_class_iou'][idx]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["convnext_unet", "segformer"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=".",
                         help="Cartella dove salvare eval_results_<model>.json e la confusion matrix")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.model, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint caricato: epoch {checkpoint['epoch']}, val_mIoU salvato={checkpoint['val_miou']:.4f}")

    _, val_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size, img_size=args.img_size,
    )

    results = evaluate_model(model, val_loader, device)
    efficiency = measure_efficiency(model, device, img_size=args.img_size)

    print_report(results, efficiency, args.model)

    # Salva la matrice di confusione per eventuali visualizzazioni nel report/notebook
    confmat_path = Path(args.output_dir) / f"confusion_matrix_{args.model}.npy"
    np.save(confmat_path, results["confusion_matrix"])
    print(f"\nMatrice di confusione salvata in: {confmat_path}")

    # Salva tutte le metriche in un JSON unico, riproducibile e citabile nel report finale
    results_path = save_results(
        results, efficiency, args.model, args.checkpoint, checkpoint["epoch"],
        output_dir=args.output_dir,
    )
    print(f"Risultati completi salvati in: {results_path}")


if __name__ == "__main__":
    main()
