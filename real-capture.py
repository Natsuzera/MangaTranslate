import cv2
import numpy as np
from ultralytics import YOLO
import mss
import time

# Carregue o seu modelo YOLO (ex: yolov8n.pt, ou o caminho para o seu .pt)
# Substitua 'yolov8n.pt' pelo caminho do seu modelo, seja ele qual for.
model = YOLO(r"K:\projetos\MangaTranslate\output\primeiro_treinamento\file_tuning.pt")

# ----- OU, para capturar uma janela específica (RECOMENDADO) -----
# Você precisa descobrir as coordenadas da janela. Execute o script e posicione a janela.
# Exemplo para uma janela de navegador no canto superior esquerdo:
monitor = {"top": 25, "left": 20, "width": 900, "height": 1000}


print("Iniciando a detecção na tela. Pressione 'q' na janela de visualização para sair.")

# Usamos o 'with' para garantir que os recursos sejam liberados corretamente
with mss.mss() as sct:
    while True:
        # Captura a imagem da área definida
        sct_img = sct.grab(monitor)
        
        # Converte a imagem capturada (formato MSS) para um array NumPy
        # MSS captura em BGRA, então precisamos converter para BGR para o OpenCV
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Realiza a detecção com o YOLO
        results = model(frame, stream=True) # Usar stream=True para melhor performance em vídeo

        # Processa os resultados
        for r in results:
            # O método plot() já desenha as caixas e rótulos na imagem
            frame = r.plot()

        # Exibe o frame com as detecções
        cv2.imshow('Detecção de Objetos na Tela (YOLO)', frame)

        # Verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Fecha todas as janelas do OpenCV
cv2.destroyAllWindows()
print("Detecção encerrada.")