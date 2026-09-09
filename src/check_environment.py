"""
Script di verifica ambiente per il progetto di Segmentazione Semantica (SII).

Verifica che:
- PyTorch veda correttamente la GPU (RTX 5060)
- CUDA sia disponibile e funzionante
- Le librerie principali siano installate con le versioni corrette

Uso:
    python src/check_environment.py
"""

import sys


def check_torch_gpu():
    print("=" * 60)
    print("1. PyTorch + GPU")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("[ERRORE] PyTorch non installato. Esegui: pip install -r requirements.txt")
        return False

    print(f"PyTorch versione: {torch.__version__}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[ATTENZIONE] CUDA non disponibile: il training girerebbe su CPU (molto lento).")
        print("Verifica i driver NVIDIA con: nvidia-smi")
        return False

    print(f"CUDA versione (PyTorch): {torch.version.cuda}")
    print(f"Numero GPU rilevate: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU {i}: {props.name} — {vram_gb:.1f} GB VRAM")

    # Test rapido: alloca un tensore su GPU
    try:
        x = torch.rand(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("Test allocazione/calcolo su GPU: OK")
    except Exception as e:
        print(f"[ERRORE] Test GPU fallito: {e}")
        return False

    # Verifica supporto mixed precision (bf16), utile con 8GB VRAM
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"Supporto bfloat16 (mixed precision): {bf16_supported}")

    return True


def check_libraries():
    print("\n" + "=" * 60)
    print("2. Librerie principali")
    print("=" * 60)

    libs = [
        "torchvision",
        "transformers",
        "segmentation_models_pytorch",
        "timm",
        "albumentations",
        "torchmetrics",
        "wandb",
        "pytorch_grad_cam",
        "cv2",
        "fastapi",
    ]

    all_ok = True
    for lib in libs:
        try:
            mod = __import__(lib)
            version = getattr(mod, "__version__", "versione non esposta")
            print(f"  [OK] {lib}: {version}")
        except ImportError:
            print(f"  [MANCANTE] {lib} — installa con: pip install -r requirements.txt")
            all_ok = False

    return all_ok


def main():
    gpu_ok = check_torch_gpu()
    libs_ok = check_libraries()

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    if gpu_ok and libs_ok:
        print("Ambiente pronto per il training.")
        sys.exit(0)
    else:
        print("Ci sono problemi da risolvere prima di iniziare il training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
