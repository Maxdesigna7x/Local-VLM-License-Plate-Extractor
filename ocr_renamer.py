import base64
import requests
import shutil
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURACIÓN PRINCIPAL
# ==========================================

# Elige tu backend: "lmstudio" o "ollama"
BACKEND = "lmstudio"  

# Nombre del modelo (Asegúrate de que coincida con el que tienes descargado)
# Nota para Ollama: A veces los modelos de visión se llaman diferente, ej: "qwen2-vl" o "llava"
MODEL_NAME = "qwen3.5-0.8b" 

# Endpoints por defecto
URL_LMSTUDIO = "http://localhost:1234/v1/chat/completions"
URL_OLLAMA = "http://localhost:11434/api/chat"

# Carpetas de trabajo
INPUT_FOLDER = "Placas_Input"
OUTPUT_FOLDER = "Dataset/RenameWebScrap"

# Crear carpeta de salida si no existe
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ==========================================
# 🧠 PROMPTING (Optimizado para Zero-Shot OCR)
# ==========================================

SYSTEM_PROMPT = """You are a specialized OCR tool for vehicle license plates.
Your ONLY task is to extract the license plate number from the image.

Rules:
1. Return ONLY the alphanumeric characters of the plate.
2. No spaces, no dashes, no extra text.
3. If no plate is found, return 'NOT_FOUND'.
4. Do not explain anything."""

USER_PROMPT = "Extract the plate number."

# ==========================================
# 🛠️ FUNCIONES AUXILIARES
# ==========================================

def encode_image(image_path):
    """Convierte la imagen a base64 para poder enviarla por JSON."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_plate(image_path):
    """Envía la imagen al VLM local y retorna el texto extraído."""
    img_base64 = encode_image(image_path)

    if BACKEND == "lmstudio":
        # Formato compatible con la API de OpenAI (LM Studio)
        payload = {
            "model": MODEL_NAME,
            "temperature": 0.0, # 0 = Respuestas deterministas y precisas
            "max_tokens": 15,   # Evita que el modelo empiece a "hablar" de más
            "stop": ["\n", " "],
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
        }
        api_url = URL_LMSTUDIO

    elif BACKEND == "ollama":
        # Formato nativo de la API de Ollama
        payload = {
            "model": MODEL_NAME,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 15,
                "stop": ["\n", " "]
            },
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT,
                    "images": [img_base64] # Ollama toma el base64 crudo en un array
                }
            ]
        }
        api_url = URL_OLLAMA
    else:
        raise ValueError("Backend no soportado. Usa 'lmstudio' u 'ollama'.")

    # Realizar la petición HTTP
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status() # Lanza error si el código HTTP no es 200 OK
        result = response.json()
        
        # Extraer texto dependiendo del backend
        if BACKEND == "lmstudio":
            plate_text = result["choices"][0]["message"]["content"]
        else: # ollama
            plate_text = result["message"]["content"]
            
        # Limpiar residuos y asegurar mayúsculas
        return plate_text.strip().upper()
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error de conexión: ¿Está {BACKEND} ejecutándose?")
        return "ERROR_CONEXION"
    except Exception as e:
        print(f"❌ Error en API para {Path(image_path).name}: {e}")
        return "ERROR"

# ==========================================
# 🚀 FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ==========================================

def process_plates(folder):
    """Itera sobre la carpeta de entrada, extrae la placa y renombra el archivo."""
    print(f"Iniciando procesamiento con el backend: {BACKEND.upper()}")
    print("-" * 50)
    
    # Busca imágenes comunes
    input_path = Path(folder)
    images = [p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    if not images:
        print(f"⚠️ No se encontraron imágenes en la carpeta '{folder}'.")
        return

    for img in images:
        try:
            plate_id = extract_plate(img)
            
            # Filtro básico de seguridad para nombres de archivo en Windows/Linux
            safe_plate_id = "".join([c for c in plate_id if c.isalnum() or c in ['_', '-']])
            
            if not safe_plate_id:
                safe_plate_id = "INVALID_OUTPUT"

            print(f"📷 Original: {img.name}  -->  📝 Extraído: {safe_plate_id}")

            # Construir el nuevo nombre: PLACA_nombreoriginal.ext
            new_name = f"{safe_plate_id}_{img.name}"
            dest = Path(OUTPUT_FOLDER) / new_name
            
            # Copiar en lugar de mover por seguridad (puedes cambiar a shutil.move si quieres)
            shutil.copy(img, dest)
            
        except Exception as e:
            print(f"❌ Fallo crítico procesando {img.name}: {e}")

# ==========================================
# ▶️ EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    process_plates(INPUT_FOLDER)
    
    # Si solo quieres probar con una imagen específica, comenta la línea de arriba y usa esta:
    # print(extract_plate("Placas_Input/test_image.jpg"))