import cv2
import numpy as np
from ultralytics import YOLO
import mss
import time
import ollama
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import hashlib

class MangaTranslator:
    def __init__(self, model_path, ollama_model="gemma3:12b", target_language="português"):
        """
        Inicializa o tradutor de mangá otimizado
        
        Args:
            model_path: Caminho para o modelo YOLO treinado
            ollama_model: Nome do modelo Ollama para OCR + tradução
            target_language: Idioma de destino para tradução
        """
        self.yolo_model = YOLO(model_path)
        self.ollama_model = ollama_model
        self.target_language = target_language
        
        # Configurações de display
        self.monitor = {"top": 25, "left": 20, "width": 900, "height": 1000}
        
        # Sistema de cache mais robusto
        self.translation_cache = {}
        self.processing_baloons = set()  # Balões sendo processados
        self.stable_detections = {}  # Detecções estáveis
        self.frame_counter = 0
        self.stability_threshold = 5  # Frames para considerar detecção estável
        
        # Queue para processamento assíncrono (menor para evitar acúmulo)
        self.processing_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        
        # Thread pool para processamento OCR + tradução
        self.processing_threads = []
        for i in range(2):  # 2 threads para processar em paralelo
            thread = threading.Thread(target=self._process_translations, daemon=True)
            thread.start()
            self.processing_threads.append(thread)
        
        # Prompt otimizado para o modelo multimodal
        self.ocr_prompt = f"""
Você é um especialista em tradução de mangás e balões de fala.

Sua tarefa é analisar a imagem fornecida (um balão de fala de mangá) e seguir os passos abaixo:

1. Identifique e transcreva todo o texto visível na imagem (mesmo se estiver incompleto ou difícil de ler).
2. Interprete o contexto e o significado da fala, considerando nuances culturais, expressões idiomáticas e tom.
3. Forneça APENAS a tradução final para {self.target_language}, de forma natural e fiel ao tom original.

Não inclua o texto original, explicações ou observações. Apenas a tradução final.
"""
        
        # Métricas de performance
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        print(f"Sistema inicializado com modelo: {ollama_model}")
        print(f"Idioma de destino: {target_language}")

    def _generate_stable_hash(self, bbox, frame_crop):
        """Gera hash estável baseado na posição e conteúdo da imagem"""
        # Hash baseado na posição (arredondada) e conteúdo da imagem
        x1, y1, x2, y2 = map(int, bbox)
        
        # Arredondar posição para tolerar pequenas variações
        pos_hash = f"{x1//10}_{y1//10}_{(x2-x1)//10}_{(y2-y1)//10}"
        
        # Hash do conteúdo da imagem (redimensionada para acelerar)
        small_crop = cv2.resize(frame_crop, (50, 50))
        img_hash = hashlib.md5(small_crop.tobytes()).hexdigest()[:8]
        
        return f"{pos_hash}_{img_hash}"

    def _image_to_base64(self, image_array):
        """Converte array numpy para base64 - otimizado com melhor pré-processamento"""
        try:
            # Etapa 1: Converter para escala de cinza se necessário
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array.copy()
            
            # Etapa 2: Melhorar contraste e reduzir ruído
            # Aplicar filtro bilateral para reduzir ruído mantendo bordas
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Etapa 3: Melhorar contraste usando CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Etapa 4: Aplicar threshold adaptativo para melhor binarização
            # Usar threshold adaptativo que se ajusta às condições locais
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Etapa 5: Operações morfológicas para limpar o texto
            # Kernel pequeno para preservar detalhes do texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            
            # Closing para conectar partes quebradas das letras
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Opening para remover ruído pequeno
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Etapa 6: Redimensionar mantendo proporção se necessário
            height, width = cleaned.shape
            max_dimension = 512  # Aumentar resolução máxima para melhor OCR
            
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Etapa 7: Adicionar margem branca ao redor da imagem
            margin = 10
            padded = cv2.copyMakeBorder(
                cleaned, margin, margin, margin, margin, 
                cv2.BORDER_CONSTANT, value=255
            )
            
            # Etapa 8: Converter para RGB (PIL espera RGB)
            image_rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
            
            # Etapa 9: Converter para PIL e salvar como base64
            pil_image = Image.fromarray(image_rgb)
            
            # Converter para base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG', optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            print(f"❌ Erro no pré-processamento da imagem: {e}")
            # Fallback para método simples se houver erro
            if len(image_array.shape) == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_str

    def _process_translations(self):
        """Thread worker otimizado para processamento OCR + tradução"""
        while True:
            try:
                # Pega item da queue com timeout
                crop_data = self.processing_queue.get(timeout=1)
                if crop_data is None:
                    break
                
                crop_image, stable_id, bbox = crop_data
                
                print(f"🔄 Processando balão: {stable_id}")
                
                try:
                    # Converter para base64
                    start_time = time.time()
                    img_base64 = self._image_to_base64(crop_image)
                    
                    # Processar com Ollama
                    response = ollama.chat(
                        model=self.ollama_model,
                        messages=[
                            {
                                'role': 'system',
                                'content': 'Você é um especialista em tradução de mangás com conhecimento profundo sobre cultura japonesa, expressões idiomáticas e contexto emocional.'
                            },
                            {
                                'role': 'user',
                                'content': self.ocr_prompt,
                                'images': [img_base64]
                            }
                        ],
                        options={
                            'keep_alive': 60,  # Manter modelo carregado por mais tempo
                            'num_ctx': 4096,   # Contexto maior para melhor compreensão
                            'temperature': 0.2, # Baixa criatividade, foco na precisão
                            'top_p': 0.9,      # Maior diversidade nas escolhas de palavras
                            'repeat_penalty': 1.1  # Evitar repetições
                        }
                    )
                    
                    processing_time = time.time() - start_time
                    translated_text = response['message']['content'].strip()
                    
                    # Limpar e processar texto traduzido
                    translated_text = ' '.join(translated_text.split())
                    
                    # Remover possíveis prefixos/sufixos desnecessários
                    prefixes_to_remove = [
                        "tradução:", "traduzido:", "texto:", "resultado:",
                        "português:", "em português:", "translation:",
                        "o texto diz:", "a tradução é:", "tradução final:"
                    ]
                    
                    text_lower = translated_text.lower()
                    for prefix in prefixes_to_remove:
                        if text_lower.startswith(prefix):
                            translated_text = translated_text[len(prefix):].strip()
                            break
                    
                    # Remover aspas desnecessárias
                    if translated_text.startswith('"') and translated_text.endswith('"'):
                        translated_text = translated_text[1:-1].strip()
                    if translated_text.startswith("'") and translated_text.endswith("'"):
                        translated_text = translated_text[1:-1].strip()
                    
                    # Validar se a tradução não está vazia
                    if not translated_text or len(translated_text.strip()) < 2:
                        translated_text = "[Texto não detectado]"
                    
                    # Armazenar no cache
                    self.translation_cache[stable_id] = translated_text
                    
                    print(f"✅ Traduzido em {processing_time:.2f}s: '{translated_text[:50]}...'")
                    
                    # Coloca resultado na queue
                    self.result_queue.put((stable_id, translated_text))
                    
                except Exception as e:
                    print(f"❌ Erro no OCR/tradução para {stable_id}: {e}")
                    error_text = "[Erro na tradução]"
                    self.translation_cache[stable_id] = error_text
                    self.result_queue.put((stable_id, error_text))
                
                finally:
                    # Remover da lista de processamento
                    self.processing_baloons.discard(stable_id)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erro crítico no processamento: {e}")

    def _extract_text_region(self, frame, bbox):
        """Extrai região do texto com melhor pré-processamento"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validar coordenadas
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Adicionar margem pequena
        margin = 3
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        crop = frame[y1:y2, x1:x2]
        
        # Validar crop
        if crop.size == 0:
            return None
            
        return crop

    def _is_detection_stable(self, bbox, stable_id):
        """Verifica se uma detecção é estável por vários frames"""
        if stable_id not in self.stable_detections:
            self.stable_detections[stable_id] = {
                'count': 1,
                'bbox': bbox,
                'first_seen': self.frame_counter
            }
            return False
        else:
            self.stable_detections[stable_id]['count'] += 1
            self.stable_detections[stable_id]['bbox'] = bbox
            
            # Considerar estável após threshold frames
            return self.stable_detections[stable_id]['count'] >= self.stability_threshold

    def _wrap_text(self, text, font, max_width):
        """Quebra texto em linhas de forma mais inteligente"""
        if not text:
            return []
            
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * 8  # Fallback
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Palavra muito longa, forçar quebra
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def _draw_translation(self, frame, bbox, text):
        """Desenha tradução com melhor visual"""
        if not text or text == "[Erro na tradução]":
            return frame
        
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        
        # Pular se área muito pequena
        if width < 20 or height < 20:
            return frame
        
        try:
            # Converter para PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Criar overlay transparente
            overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Calcular tamanho da fonte baseado no tamanho do balão (melhorado)
            # Usar tamanho mais adequado baseado na área do balão
            area = width * height
            if area > 10000:  # Balões grandes
                font_size = min(24, max(16, min(width//6, height//3)))
            elif area > 5000:  # Balões médios
                font_size = min(20, max(14, min(width//7, height//3.5)))
            else:  # Balões pequenos
                font_size = min(16, max(12, min(width//8, height//4)))
            
            # Tentar carregar a fonte AnimeAce2 primeiro
            try:
                font = ImageFont.truetype("animeace2_bld.ttf", font_size)
            except:
                # Fallback para Arial se AnimeAce2 não funcionar
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Fundo branco semi-transparente com bordas mais suaves
            draw.rectangle([x1, y1, x2, y2], 
                         fill=(255, 255, 255, 220), 
                         outline=(0, 0, 0, 180), 
                         width=2)
            
            # Quebrar texto
            lines = self._wrap_text(text, font, width - 12)  # Mais margem para texto
            
            if lines:
                # Calcular posição centralizada
                line_height = font_size + 4  # Mais espaçamento entre linhas
                total_height = len(lines) * line_height
                start_y = y1 + (height - total_height) // 2
                
                # Desenhar cada linha com sombra para melhor legibilidade
                for i, line in enumerate(lines):
                    try:
                        bbox_line = font.getbbox(line)
                        text_width = bbox_line[2] - bbox_line[0]
                    except:
                        text_width = len(line) * (font_size // 2)
                    
                    text_x = x1 + (width - text_width) // 2
                    text_y = start_y + i * line_height
                    
                    # Desenhar sombra do texto (offset de 1 pixel)
                    draw.text((text_x + 1, text_y + 1), line, fill=(128, 128, 128, 180), font=font)
                    
                    # Desenhar texto principal
                    draw.text((text_x, text_y), line, fill=(0, 0, 0, 255), font=font)
            
            # Combinar com imagem original
            result = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
            return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"❌ Erro ao desenhar tradução: {e}")
            return frame

    def _calculate_fps(self):
        """Calcula FPS atual"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def _cleanup_old_detections(self):
        """Remove detecções antigas que não aparecem mais"""
        current_frame = self.frame_counter
        to_remove = []
        
        for stable_id, data in self.stable_detections.items():
            if current_frame - data['first_seen'] > 60:  # 60 frames sem aparecer
                to_remove.append(stable_id)
        
        for stable_id in to_remove:
            del self.stable_detections[stable_id]
            self.translation_cache.pop(stable_id, None)

    def run(self):
        """Executa o sistema de tradução otimizado"""
        print("🚀 Iniciando sistema de tradução de mangá otimizado...")
        print("Controles:")
        print("  'q' - Sair")
        print("  'c' - Limpar cache")
        print("  'r' - Resetar detecções")
        print("  'p' - Pausar/Retomar processamento")
        print("=" * 50)
        
        paused = False
        
        with mss.mss() as sct:
            while True:
                self.frame_counter += 1
                
                # Captura da tela
                sct_img = sct.grab(self.monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Detecção YOLO
                results = self.yolo_model(frame, stream=True, verbose=False)
                
                display_frame = frame.copy()
                
                # Processar detecções apenas se não pausado
                if not paused:
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None:
                            for i, box in enumerate(boxes):
                                bbox = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                
                                # Filtrar detecções com baixa confiança
                                if confidence < 0.6:
                                    continue
                                
                                # Extrair região
                                text_region = self._extract_text_region(frame, bbox)
                                if text_region is None:
                                    continue
                                
                                # Gerar ID estável
                                stable_id = self._generate_stable_hash(bbox, text_region)
                                
                                # Verificar se detecção é estável
                                if self._is_detection_stable(bbox, stable_id):
                                    # Desenhar bounding box
                                    x1, y1, x2, y2 = map(int, bbox)
                                    
                                    # Verificar se já temos tradução
                                    if stable_id in self.translation_cache:
                                        # Desenhar tradução
                                        display_frame = self._draw_translation(
                                            display_frame, bbox, self.translation_cache[stable_id]
                                        )
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(display_frame, f"✓ {confidence:.2f}", 
                                                  (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.4, (0, 255, 0), 1)
                                    else:
                                        # Verificar se já está sendo processado
                                        if stable_id not in self.processing_baloons:
                                            # Adicionar à queue de processamento
                                            try:
                                                self.processing_queue.put((text_region, stable_id, bbox), block=False)
                                                self.processing_baloons.add(stable_id)
                                                print(f"📥 Adicionado à queue: {stable_id}")
                                            except queue.Full:
                                                print("⚠️ Queue cheia, pulando este balão")
                                        
                                        # Mostrar status
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        cv2.putText(display_frame, f"🔄 {confidence:.2f}", 
                                                  (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.4, (0, 255, 255), 1)
                                else:
                                    # Detecção instável
                                    x1, y1, x2, y2 = map(int, bbox)
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                
                # Coletar resultados de tradução
                try:
                    while True:
                        stable_id, translated_text = self.result_queue.get_nowait()
                        print(f"✅ Recebido resultado: {stable_id} -> '{translated_text[:30]}...'")
                except queue.Empty:
                    pass
                
                # Limpar detecções antigas periodicamente
                if self.frame_counter % 60 == 0:
                    self._cleanup_old_detections()
                
                # Calcular FPS
                self._calculate_fps()
                
                # Informações na tela
                info_y = 25
                cv2.putText(display_frame, f"FPS: {self.current_fps}", 
                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Cache: {len(self.translation_cache)}", 
                          (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Processando: {len(self.processing_baloons)}", 
                          (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Queue: {self.processing_queue.qsize()}", 
                          (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if paused:
                    cv2.putText(display_frame, "PAUSADO", 
                              (10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Exibir resultado
                cv2.imshow('Manga Translator - Otimizado', display_frame)
                
                # Controles do teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.translation_cache.clear()
                    self.processing_baloons.clear()
                    print("🧹 Cache limpo!")
                elif key == ord('r'):
                    self.stable_detections.clear()
                    self.translation_cache.clear()
                    self.processing_baloons.clear()
                    print("🔄 Sistema resetado!")
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️ Processamento {'pausado' if paused else 'retomado'}")
        
        # Cleanup
        for _ in range(len(self.processing_threads)):
            self.processing_queue.put(None)
        
        cv2.destroyAllWindows()
        print("🛑 Sistema encerrado.")

def main():
    # Configurações
    model_path = r"K:\projetos\MangaTranslate\output\primeiro_treinamento\file_tuning.pt"
    ollama_model = "qwen2.5vl:7b"  # ou "gemma3:4b-it-qat"
    target_language = "português"
    
    # Criar e executar tradutor
    translator = MangaTranslator(model_path, ollama_model, target_language)
    translator.run()

if __name__ == "__main__":
    main()