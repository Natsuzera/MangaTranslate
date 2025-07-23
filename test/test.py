import cv2
import os
from ultralytics import YOLO
import torch
from pathlib import Path


def setup_directories():
    """Cria a pasta de output se não existir"""
    output_dir = Path("../output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_yolo_model():
    """Carrega o modelo YOLO treinado"""
    # Verifica se existe modelo treinado, senão usa modelo base
    custom_model_path = "../output/primeiro_treinamento/file_tuning.pt"
    fallback_model_path = "../output/primeiro_treinamento/file_tuning.pt"
    base_model_path = "../yolo11n.pt"
    
    if os.path.exists(custom_model_path):
        print(f"Carregando modelo customizado: {custom_model_path}")
        model = YOLO(custom_model_path)
    elif os.path.exists(fallback_model_path):
        print(f"Carregando modelo fallback: {fallback_model_path}")
        model = YOLO(fallback_model_path)
    elif os.path.exists(base_model_path):
        print(f"Carregando modelo base: {base_model_path}")
        model = YOLO(base_model_path)
    else:
        print("Nenhum modelo encontrado! Baixando YOLO11n...")
        model = YOLO("yolo11n.pt")
    
    return model


def test_detection_on_image(model, image_path, output_dir, confidence_threshold=0.25):
    """
    Realiza detecção em uma imagem e salva o resultado
    
    Args:
        model: Modelo YOLO carregado
        image_path: Caminho para a imagem de teste
        output_dir: Diretório de saída
        confidence_threshold: Limite de confiança para detecções
    """
    if not os.path.exists(image_path):
        print(f"❌ Imagem não encontrada: {image_path}")
        return False
    
    print(f"🔍 Processando: {os.path.basename(image_path)}")
    
    try:
        # Carrega a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erro ao carregar a imagem: {image_path}")
            return False
        
        # Realiza a detecção
        results = model(image, conf=confidence_threshold)
        
        # Processa os resultados
        annotated_image = results[0].plot()
        
        # Define o nome do arquivo de saída
        image_name = Path(image_path).stem
        output_path = output_dir / f"{image_name}_detected.jpg"
        
        # Salva a imagem com as detecções
        cv2.imwrite(str(output_path), annotated_image)
        
        # Imprime estatísticas
        detections = results[0].boxes
        if detections is not None:
            num_detections = len(detections)
            print(f"✅ {num_detections} detecções encontradas")
            
            # Mostra detalhes das detecções
            for i, box in enumerate(detections):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                print(f"   Detecção {i+1}: {class_name} (confiança: {conf:.2f})")
        else:
            print("✅ Nenhuma detecção encontrada")
        
        print(f"💾 Resultado salvo em: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erro durante a detecção: {str(e)}")
        return False


def main():
    """Função principal do teste"""
    print("🚀 Iniciando teste de detecção YOLO")
    print("=" * 50)
    
    # Verifica disponibilidade da GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU disponível: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("💻 Usando CPU")
        device = "cpu"
    
    print()
    
    # Configura diretórios
    output_dir = setup_directories()
    print(f"📁 Pasta de saída: {output_dir.absolute()}")
    
    # Carrega o modelo
    print("\n🤖 Carregando modelo YOLO...")
    model = load_yolo_model()
    model.to(device)
    
    # Lista de imagens para testar
    test_images = [
        "../test1.png",
        "../teste.png"
    ]
    
    # Adiciona test2.png se existir
    if os.path.exists("../test2.png"):
        test_images.append("../test2.png")
    
    print(f"\n🖼️  Imagens para teste: {len(test_images)}")
    
    # Processa cada imagem
    successful_tests = 0
    for image_path in test_images:
        print("\n" + "-" * 30)
        success = test_detection_on_image(model, image_path, output_dir)
        if success:
            successful_tests += 1
    
    # Resumo final
    print("\n" + "=" * 50)
    print(f"📊 RESUMO: {successful_tests}/{len(test_images)} imagens processadas com sucesso")
    
    if successful_tests > 0:
        print(f"✅ Resultados salvos em: {output_dir.absolute()}")
    
    print("🏁 Teste concluído!")


if __name__ == "__main__":
    main()