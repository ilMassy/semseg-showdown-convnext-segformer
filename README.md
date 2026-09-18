# 🧠 Segmentazione Semantica Comparata: CNN vs Transformer

> Confronto tra un'architettura encoder-decoder convoluzionale moderna (**ConvNeXt**)
> e un'architettura transformer-based nativa per la segmentazione (**SegFormer**),
> valutate sul dataset **PASCAL VOC 2012 (Segmentation)**.

| | |
|---|---|
| 🖼️ **Dataset** | [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) (Segmentation) |
| 🖥️ **Hardware** | GPU dedicata, esecuzione locale |
| 🐧 **Sistema operativo** | Unix-based (Linux) |
| 🧩 **Modelli** | ConvNeXt-UNet · SegFormer |

---

## 📖 Descrizione

L'obiettivo è realizzare un sistema di segmentazione semantica delle immagini,
con un confronto approfondito tra architetture convoluzionali moderne e
architetture transformer-based, orientato non solo alle performance predittive
ma anche al rigore metodologico e all'analisi critica dei risultati. Il progetto
nasce nell'ambito del corso di Sistemi Intelligenti per Internet.

Il progetto copre:

- 🏗️ **Confronto architetturale** — ConvNeXt vs SegFormer (PyTorch, HuggingFace Transformers)
- 📊 **Valutazione rigorosa** — mIoU, Dice score, F1 per classe
- 🔍 **Analisi degli errori e interpretabilità** — Grad-CAM / attention map
- 🧪 **Ablation study** — augmentation, loss functions, risoluzione input
- ⚡ **Analisi di efficienza** — tempi di training/inferenza, numero di parametri

Tutta l'infrastruttura (training, inferenza, valutazione) viene eseguita in
locale su GPU dedicata, senza servizi cloud a pagamento.

---

## 🗂️ Struttura del progetto

```
segmentation-project/
├── data/               # Dataset (scaricato, non versionato su Git)
├── src/                # Codice sorgente
│   ├── check_environment.py   # Verifica GPU/CUDA/librerie
│   ├── download_dataset.sh    # Download PASCAL VOC 2012
│   ├── dataset.py              # Dataset + augmentation (Albumentations)
│   ├── models.py                # ConvNeXt-UNet e SegFormer
│   ├── train.py                  # Training loop (loss, augmentation, efficienza)
│   ├── evaluate.py               # Metriche per classe (mIoU, Dice/F1, confusion matrix)
│   └── explain.py                # Grad-CAM (ConvNeXt) / attention map (SegFormer)
├── results/             # Output di evaluate.py ed explain.py
│   ├── eval_results_convnext_unet.json   # Metriche per classe + efficienza
│   ├── eval_results_segformer.json       # Metriche per classe + efficienza
│   ├── confusion_matrix_convnext_unet.npy  # Confusion matrix 21x21
│   ├── confusion_matrix_segformer.npy      # Confusion matrix 21x21
│   └── explanations/    # Grad-CAM / attention map su esempi reali
├── notebooks/          # Analisi esplorativa, visualizzazioni
├── checkpoints/         # Pesi dei modelli salvati (non versionato su Git)
├── configs/             # File di configurazione esperimenti (YAML)
├── requirements.txt
└── README.md
```

---

## ⚠️ Requisiti per la replica

> Questo progetto è pensato per essere eseguito su un **sistema operativo Unix-based**
> (Linux). Gli script di setup e download (`.sh`) non sono compatibili
> nativamente con Windows: su quella piattaforma è necessario un ambiente
> compatibile come **WSL** (Windows Subsystem for Linux).

## ⚙️ Setup ambiente

```bash
# 1. Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Verifica che tutto funzioni (GPU, CUDA, librerie)
python src/check_environment.py
```

## 📥 Download dataset

```bash
bash src/download_dataset.sh
```

Nessuna registrazione o attesa di approvazione richiesta: il download è immediato.

## 📦 Checkpoint pretrained

I pesi dei modelli addestrati (`checkpoints/`) non sono versionati su Git (uno dei due file
supera il limite di 100MB di GitHub) — sono ospitati su Hugging Face Hub:
👉 [ilMassy/semseg-convnext-segformer-voc2012](https://huggingface.co/ilMassy/semseg-convnext-segformer-voc2012)

Per scaricarli e rimetterli al posto giusto prima di lanciare `evaluate.py` o `explain.py`:

```bash
mkdir -p checkpoints
wget https://huggingface.co/ilMassy/semseg-convnext-segformer-voc2012/resolve/main/convnext_unet_best.pt -O checkpoints/convnext_unet_best.pt
wget https://huggingface.co/ilMassy/semseg-convnext-segformer-voc2012/resolve/main/segformer_best.pt -O checkpoints/segformer_best.pt
```

---

## 🗺️ Roadmap

- [x] Setup ambiente e struttura progetto (venv, dipendenze, verifica GPU)
  - *Nota:* Verificato con `check_environment.py` (PyTorch, CUDA compatibile e librerie principali installate).
- [x] Download e verifica dataset PASCAL VOC 2012
  - *Nota:* Dataset scaricato, estratto in `data/VOCdevkit/VOC2012` e validato (17.125 immagini JPEG, 2.913 maschere).
- [x] `dataset.py` — caricamento dataset + pipeline di augmentation (Albumentations)
  - *Nota:* Testato con smoke test (`python src/dataset.py`), output verificato: shape corrette, void index (255) gestito, split train/val standard (1464/1449 immagini). Augmentation disattivabile (`use_augmentation`) per l'ablation study.
- [x] `models.py` — ConvNeXt-UNet (`segmentation-models-pytorch`) e SegFormer (`transformers`)
  - *Nota:* Testato con smoke test (`python src/models.py`), entrambi i modelli producono output [2, 21, 512, 512]. Parametri totali: ConvNeXt-UNet 31.93M, SegFormer 3.72M — dato utile per l'analisi di efficienza.
- [x] `train.py` — training loop con mixed precision, gradient accumulation, checkpointing
  - *Nota:* Training completo eseguito per entrambi i modelli (50 epoche, `dice_focal`, augmentation attiva, `img_size=512`). ConvNeXt-UNet: mIoU=0.7594, Dice=0.8549, 67.6min. SegFormer: mIoU=0.6543, Dice=0.7803, 35.0min. Loss selezionabile (`--loss_type ce/dice_focal`) e augmentation disattivabile (`--no_augmentation`) per l'ablation study.
- [x] `evaluate.py` — mIoU, Dice/F1 per classe, matrice di confusione
  - *Nota:* Eseguito su entrambi i checkpoint. Classi più deboli per entrambi i modelli: `chair`, `sofa`, `diningtable` (difficoltà intrinseca del dataset). Risultati salvati in `eval_results_<model>.json`.
- [x] `explain.py` — Grad-CAM per ConvNeXt, attention map per SegFormer
  - *Nota:* Confermato funzionante su GPU reale per entrambi i modelli, sia in modalità base (target = predizione) sia in modalità "analisi errori" (target = ground truth, `--mask_path`).
- [ ] Ablation study — augmentation, loss (Cross-Entropy vs Dice+Focal), risoluzione input
- [x] Analisi di efficienza — tempi di training/inferenza, numero di parametri
  - *Nota:* Training: 67.6min/50 epoche ConvNeXt-UNet, 35.0min SegFormer. Inferenza (batch=1): 21.38ms/46.78 FPS ConvNeXt-UNet, 7.48ms/133.63 FPS SegFormer.
- [ ] Report finale + repository GitHub pubblico

---

## 📊 Metriche di valutazione

| Metrica | Perché |
|---|---|
| **mIoU** | Standard per la segmentazione, penalizza sia falsi positivi che negativi |
| **Dice score** | Sensibile a classi sbilanciate |
| **F1 per classe** | Evidenzia il comportamento su classi minoritarie |

---

## 👥 Autore

Massimiliano Giangreco
