"""
Interpretabilità dei due modelli (punto 3 della proposta al professore):

- ConvNeXt-UNet -> Grad-CAM sull'ultimo blocco convoluzionale dell'encoder,
  usando SemanticSegmentationTarget per calcolare il Grad-CAM rispetto a una
  classe specifica (tecnica standard per estendere Grad-CAM alla segmentazione,
  vedi libreria pytorch-grad-cam).
- SegFormer -> visualizzazione delle attention map medie dell'ultimo layer
  transformer, che mostrano su quali regioni dell'immagine il modello concentra
  l'attenzione per la predizione. Nota: a differenza del Grad-CAM di ConvNeXt,
  questa mappa NON è class-conditioned (limite architetturale dell'attenzione
  globale di SegFormer) — va dichiarato esplicitamente nel report finale.

L'obiettivo non è solo produrre immagini "belle", ma usarle per l'analisi degli
errori: confrontare dove i due modelli guardano quando sbagliano su una classe.

Modalità "analisi errori" (--mask_path): se si fornisce la maschera ground-truth
VOC (da data/VOCdevkit/VOC2012/SegmentationClass/), il Grad-CAM di ConvNeXt-UNet
usa la regione REALE della classe target (non quella predetta) — così anche
quando il modello manca completamente la classe (es. chair, sofa, diningtable,
le più deboli per entrambi i modelli, vedi PROGRESS_REPORT.md §3.4), si vede
comunque cosa il modello ha "guardato" in quei pixel. Per SegFormer, in questa
modalità viene disegnato il contorno della regione ground-truth sopra l'attention
map, come riferimento visivo (l'attention resta comunque non class-conditioned).

Uso:
    # Modalità base (target = predizione del modello)
    python src/explain.py --model convnext_unet --checkpoint checkpoints/convnext_unet_best.pt --image_path data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg --target_class 1

    # Modalità analisi errori (target = ground truth), consigliata per le classi deboli
    python src/explain.py --model convnext_unet --checkpoint checkpoints/convnext_unet_best.pt --image_path data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --mask_path data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png --target_class 9

    python src/explain.py --model segformer --checkpoint checkpoints/segformer_best.pt --image_path data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, VOC_CLASSES, VOID_INDEX
from models import build_model


def voc_color_map(num_classes: int = 21) -> np.ndarray:
    """Palette ufficiale del devkit PASCAL VOC, per visualizzare le maschere con gli stessi
    colori usati nella letteratura/nei paper sul dataset (invece di 'tab20' generico, che tra
    l'altro ha solo 20 colori distinti contro le 21 classi VOC)."""
    def bitget(byteval, idx):
        return (byteval >> idx) & 1

    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap / 255.0


VOC_CMAP = ListedColormap(voc_color_map(NUM_CLASSES))


class SemanticSegmentationTarget:
    """Target per Grad-CAM in un task di segmentazione: invece di massimizzare
    lo score di una singola classe (come nella classificazione standard),
    massimizza la somma degli score della classe scelta sui pixel dove il
    modello l'ha effettivamente predetta — così il Grad-CAM evidenzia le
    regioni che hanno guidato quella specifica predizione."""

    def __init__(self, category: int, mask: np.ndarray):
        self.category = category
        self.mask = torch.from_numpy(mask)

    def __call__(self, model_output):
        mask = self.mask.to(model_output.device)
        return (model_output[self.category, :, :] * mask).sum()


def load_image(image_path: str, img_size: int = 512):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    rgb_float = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).clone()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, rgb_float


def load_ground_truth_mask(mask_path: str, img_size: int) -> np.ndarray:
    """Carica una maschera VOC (PNG a palette, valori = indice classe, 255 = void/ignore) e la
    ridimensiona con interpolazione nearest-neighbor, per non alterare i valori delle etichette
    come farebbe un resize con interpolazione bilineare/bicubica."""
    mask = Image.open(mask_path)
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
    return np.array(mask)


def region_contour_overlay(rgb_float: np.ndarray, binary_mask: np.ndarray,
                            color: tuple = (1.0, 1.0, 0.0)) -> np.ndarray:
    """Disegna il contorno di una regione binaria sopra un'immagine RGB float [0,1]. Usato per
    mostrare dove si trova DAVVERO la classe target (da ground truth) sopra a una mappa che non
    è class-conditioned, come l'attention map di SegFormer."""
    display = (rgb_float * 255).astype(np.uint8).copy()
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(display, contours, -1, tuple(int(c * 255) for c in color), 2)
    return display.astype(np.float32) / 255.0


def class_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, target_class: int,
              ignore_index: int = VOID_INDEX) -> float:
    """IoU della singola classe target su questa singola immagine — solo per dare un numero di
    contesto nella didascalia della figura qualitativa; non sostituisce le metriche aggregate
    di evaluate.py, che restano il riferimento per il report."""
    valid = gt_mask != ignore_index
    pred_bin = (pred_mask == target_class) & valid
    gt_bin = (gt_mask == target_class) & valid
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred_bin, gt_bin).sum() / union)


def explain_convnext_unet(model, image_tensor, rgb_float, target_class, device,
                           gt_binary_mask=None):
    """Grad-CAM sull'ultimo layer convoluzionale dell'encoder ConvNeXt.
    Nota: il target_layer va adattato in base al nome effettivo esposto da
    segmentation-models-pytorch/timm; qui si usa l'ultimo stage dell'encoder,
    che è tipicamente il più informativo per Grad-CAM.

    Se gt_binary_mask è fornita, il target del Grad-CAM è la regione REALE della classe
    (ground truth) invece di quella predetta — necessario per analizzare i casi in cui il
    modello manca completamente la classe (target_mask altrimenti vuota)."""
    encoder = model.model.encoder
    # NOTA: encoder.model è un timm FeatureListNet (creato da smp con
    # features_only=True), non il modello timm "pieno" — non espone un attributo
    # .stages (Sequential), ma i singoli stage come attributi separati
    # (stages_0, stages_1, ...). Prendiamo l'ultimo child in ordine di inserimento,
    # che corrisponde all'ultimo stage selezionato — approccio robusto rispetto al
    # numero esatto di stage esposti (dipende da out_indices).
    target_layers = [list(encoder.model.children())[-1]]

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    if gt_binary_mask is not None:
        target_mask = gt_binary_mask.astype(np.float32)
        mask_source = "ground truth"
    else:
        target_mask = (pred_mask == target_class).astype(np.float32)
        mask_source = "predizione"

    if target_mask.sum() == 0:
        print(f"Attenzione: la classe '{VOC_CLASSES[target_class]}' non è presente nella "
              f"maschera di riferimento ({mask_source}) per questa immagine. "
              f"Il Grad-CAM sarà poco informativo.")

    targets = [SemanticSegmentationTarget(target_class, target_mask)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)[0]

    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return cam_image, pred_mask, mask_source


def explain_segformer(model, image_tensor, device):
    """Estrae e visualizza le attention map medie dell'ultimo layer transformer
    di SegFormer, mediate su tutte le teste di attenzione."""
    inner_model = model.model.segformer  # backbone transformer dentro il wrapper HF

    # Le versioni recenti di transformers usano di default l'implementazione "sdpa"
    # (kernel fuso PyTorch), che NON calcola/restituisce i pesi di attenzione anche con
    # output_attentions=True (outputs.attentions resta una tupla vuota). Va forzata
    # esplicitamente l'implementazione "eager" prima della forward pass.
    hf_model = model.model
    if hasattr(hf_model, "set_attn_implementation"):
        hf_model.set_attn_implementation("eager")
    else:
        # Fallback per versioni più vecchie di transformers senza questo metodo
        hf_model.config._attn_implementation = "eager"
        inner_model.config._attn_implementation = "eager"

    with torch.no_grad():
        outputs = inner_model(pixel_values=image_tensor.to(device), output_attentions=True)

    if not outputs.attentions:
        raise RuntimeError(
            "Il modello non ha restituito le attention map nemmeno dopo aver forzato "
            "l'implementazione 'eager'. Verifica la versione di transformers installata "
            "(pip show transformers) — potrebbe servire un fallback diverso per versioni "
            "molto recenti o molto vecchie della libreria."
        )

    # attentions: tupla di tensori, uno per stage; prendiamo l'ultimo stage disponibile
    last_stage_attentions = outputs.attentions[-1]  # shape: [B, num_heads, N, N]
    avg_attention = last_stage_attentions.mean(dim=1).squeeze(0).cpu().numpy()  # media sulle teste

    # Attenzione media ricevuta da ogni token (colonna della matrice di attenzione)
    attention_per_token = avg_attention.mean(axis=0)
    side = int(np.sqrt(attention_per_token.shape[0]))
    attention_map = attention_per_token[: side * side].reshape(side, side)

    attention_map = cv2.resize(attention_map, (image_tensor.shape[-1], image_tensor.shape[-2]))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    return attention_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["convnext_unet", "segformer"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None,
                         help="Path alla maschera ground-truth VOC "
                              "(data/VOCdevkit/VOC2012/SegmentationClass/<id>.png). Se fornita, "
                              "il Grad-CAM di ConvNeXt-UNet usa la regione REALE della classe "
                              "target invece di quella predetta — consigliato per le classi "
                              "deboli (chair, sofa, diningtable, vedi PROGRESS_REPORT.md §3.4).")
    parser.add_argument("--target_class", type=int, default=15,
                         help="Indice classe VOC per Grad-CAM (default: 15 = person). Per "
                              "SegFormer non guida l'attention map (non class-conditioned), ma "
                              "se --mask_path è dato viene usato per disegnarne il contorno.")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="explanations",
                         help="Cartella di output; il nome file viene generato automaticamente "
                              "da modello/immagine/classe per non sovrascrivere run precedenti.")
    parser.add_argument("--output", type=str, default=None,
                         help="Nome file di output esplicito (sovrascrive la generazione automatica).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.model, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor, rgb_float = load_image(args.image_path, args.img_size)

    gt_mask, gt_binary_mask = None, None
    if args.mask_path:
        gt_mask = load_ground_truth_mask(args.mask_path, args.img_size)
        gt_binary_mask = (gt_mask == args.target_class).astype(np.float32)

    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    panel = 0
    axes[panel].imshow(rgb_float)
    axes[panel].set_title("Immagine originale")
    panel += 1
    if gt_mask is not None:
        axes[panel].imshow(gt_mask, cmap=VOC_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        axes[panel].set_title("Ground truth")
        panel += 1

    if args.model == "convnext_unet":
        cam_image, pred_mask, mask_source = explain_convnext_unet(
            model, image_tensor, rgb_float, args.target_class, device, gt_binary_mask,
        )
        axes[panel].imshow(pred_mask, cmap=VOC_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        axes[panel].set_title("Maschera predetta")
        panel += 1
        axes[panel].imshow(cam_image)
        axes[panel].set_title(f"Grad-CAM: {VOC_CLASSES[args.target_class]} ({mask_source})")

    else:  # segformer
        attention_map = explain_segformer(model, image_tensor, device)
        with torch.no_grad():
            logits = model(image_tensor.to(device))
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        axes[panel].imshow(pred_mask, cmap=VOC_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        axes[panel].set_title("Maschera predetta")
        panel += 1

        base_display = rgb_float
        title = "Attention map (ultimo layer)"
        if gt_binary_mask is not None:
            base_display = region_contour_overlay(rgb_float, gt_binary_mask)
            title += f" + contorno GT: {VOC_CLASSES[args.target_class]}"
        axes[panel].imshow(base_display)
        axes[panel].imshow(attention_map, cmap="jet", alpha=0.5)
        axes[panel].set_title(title)

    for ax in axes:
        ax.axis("off")

    if gt_mask is not None:
        iou = class_iou(pred_mask, gt_mask, args.target_class)
        fig.suptitle(f"IoU '{VOC_CLASSES[args.target_class]}' su questa immagine: {iou:.3f}")

    plt.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = output_dir / args.output
    else:
        image_name = Path(args.image_path).stem
        cls_name = VOC_CLASSES[args.target_class]
        suffix = "gt" if gt_mask is not None else "pred"
        output_path = output_dir / f"{args.model}_{image_name}_{cls_name}_{suffix}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualizzazione salvata in: {output_path}")


if __name__ == "__main__":
    main()
