#!/usr/bin/env bash
#
# Download del dataset PASCAL VOC 2012 (Segmentation) per il progetto SII.
#
# Uso (funziona da qualsiasi directory):
#   bash src/download_dataset.sh
#
set -e

# Risolve il percorso dello script stesso, così DATA_DIR punta sempre a
# <root-progetto>/data indipendentemente dalla cartella da cui viene lanciato.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TAR="${DATA_DIR}/VOCtrainval_11-May-2012.tar"

mkdir -p "${DATA_DIR}"

if [ -d "${DATA_DIR}/VOCdevkit" ]; then
    echo "Dataset già presente in ${DATA_DIR}/VOCdevkit — nulla da fare."
    exit 0
fi

echo "Scaricamento PASCAL VOC 2012 (~2GB)..."
# Mirror ufficiale può essere lento/instabile: in caso di errore, prova il mirror
# alternativo su Kaggle (https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)
# o quello reso disponibile su torchvision (vedi commento sotto).
wget -c "${VOC_URL}" -O "${VOC_TAR}"

echo "Estrazione..."
tar -xf "${VOC_TAR}" -C "${DATA_DIR}"

echo "Fatto. Struttura dataset in: ${DATA_DIR}/VOCdevkit/VOC2012"
echo ""
echo "Nota: in alternativa puoi scaricare il dataset direttamente via torchvision:"
echo "  from torchvision.datasets import VOCSegmentation"
echo "  VOCSegmentation(root='./data', year='2012', image_set='train', download=True)"
