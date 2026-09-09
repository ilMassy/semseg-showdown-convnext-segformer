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
│   ├── dataset.py              # (da creare) Dataset + augmentation
│   ├── models.py                # (da creare) ConvNeXt-UNet e SegFormer
│   ├── train.py                  # (da creare) Training loop
│   ├── evaluate.py               # (da creare) Metriche (mIoU, Dice, F1)
│   └── explain.py                # (da creare) Grad-CAM / attention map
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

---

## 🗺️ Roadmap

- [x] Setup ambiente e struttura progetto (venv, dipendenze, verifica GPU)
  - *Nota:* Verificato con `check_environment.py` (PyTorch, CUDA compatibile e librerie principali installate).
- [x] Download e verifica dataset PASCAL VOC 2012
  - *Nota:* Dataset scaricato, estratto in `data/VOCdevkit/VOC2012` e validato (17.125 immagini JPEG, 2.913 maschere).
- [ ] `dataset.py` — caricamento dataset + pipeline di augmentation (Albumentations)
- [ ] `models.py` — ConvNeXt-UNet (`segmentation-models-pytorch`) e SegFormer (`transformers`)
- [ ] `train.py` — training loop con mixed precision e logging su Weights & Biases
- [ ] `evaluate.py` — mIoU, Dice, F1 per classe, matrice di confusione
- [ ] `explain.py` — Grad-CAM per ConvNeXt, attention map per SegFormer
- [ ] Ablation study — augmentation, loss (Cross-Entropy vs Dice+Focal), risoluzione input
- [ ] Analisi di efficienza — tempi di training/inferenza, numero di parametri
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
