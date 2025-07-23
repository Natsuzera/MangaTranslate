import torch
import torchvision
from ultralytics import YOLO


def treinar_modelo():
    device = torch.device(0)

    # Carrega o modelo
    model = YOLO(r"K:\projetos\MangaTranslate\output\primeiro_treinamento\file_tuning.pt")  # Primeiro dataset treinado
    model.to(device)

    print("Iniciando o treinamento...")

    # Treinamento do modelo
   
    model.train(
        data='K:\projetos\MangaTranslate\manga109_text_detector\dataset.yaml',
        epochs=40,          # Aumentado para um treinamento mais completo
        patience=20,         # Parada antecipada para economizar tempo
        batch=8,             # Mantenha se couber na sua VRAM
        imgsz=640,           # Ótimo ponto de partida
        optimizer="AdamW",   # Boa escolha
        pretrained=True,     # Essencial, sempre use
        amp=True,            # Ótimo para velocidade
        
        # --- Augmentations Refinadas ---
        augment=True,
        degrees=5.0,         # Leves rotações, comum em scans
        translate=0.1,       # Pequenos deslocamentos de posição
        scale=0.5,           # Variação de escala (zoom) para textos de vários tamanhos
        flipud=0.0,          # DESATIVAR flip vertical, texto não aparece de cabeça para baixo
        fliplr=0.5,          # Manter flip horizontal (padrão)
        
        # Seus valores de HSV são bons para garantir robustez de cor
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    # Salva o modelo treinado
    model.save("output/primeiro_treinamento/file_tuning_mangaAdd.pt")


def main():
    print("Testando placa de vídeo e versões do PyTorch e Torchvision:")
    print("Versão do PyTorch:", torch.__version__)
    print("Versão do Torchvision:", torchvision.__version__)
    print("CUDA disponível:", torch.cuda.is_available())
    print("Placa de vídeo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")
    print("\n")

    if torch.cuda.is_available():
        print("Treinando modelo...")
        treinar_modelo()
    else:
        print("CUDA não está disponível. Treinamento não pode ser realizado.")

    





if __name__ == "__main__":
    main()
