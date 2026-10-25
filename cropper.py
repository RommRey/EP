import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO

register_heif_opener()

INPUT_FOLDER = "naturales"
OUTPUT_FOLDER = "recortes"
DISCARD_FOLDER = "descartadas"

SHARPNESS_THRESHOLD_PERCENT = 10.0
REFERENCE_LAPLACIAN_VAR = 100.0  

TARGET_CLASSES = [0, 32]  # Persona y balón
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.heic', '.heif')

model = None

def init_worker():
    """Volvemos al modelo nano ultrarrápido."""
    global model
    model = YOLO("yolov8n.pt")

def get_yolo_model():
    global model
    if model is None:
        init_worker()
    return model

def calcular_porcentaje_nitidez(pil_img):
    open_cv_image = np.array(pil_img.convert('L'))
    laplacian_var = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    percentage = (laplacian_var / REFERENCE_LAPLACIAN_VAR) * 100.0
    return min(percentage, 100.0), laplacian_var

def procesar_foto_crop(filename, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, discard_folder=DISCARD_FOLDER, threshold=SHARPNESS_THRESHOLD_PERCENT):
    img_path = os.path.join(input_folder, filename)
    
    raw_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    
    nitidez_pct, lap_var = calcular_porcentaje_nitidez(pil_img)
    
    if nitidez_pct < threshold:
        discard_path = os.path.join(discard_folder, filename)
        shutil.copy(img_path, discard_path)
        print(f"❌ Descartada (Nitidez: {nitidez_pct:.1f}%): {filename}")
        return

    print(f"✅ Pasa filtro (Nitidez: {nitidez_pct:.1f}%): Recortando {filename}...")
    w_orig, h_orig = pil_img.size

    yolo_net = get_yolo_model()
    
    # Inferencia optimizada a 640px para mantener velocidad
    results = yolo_net(pil_img, imgsz=640, conf=0.25, verbose=False)[0]
    
    action_boxes = []
    ball_centers = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            
            if cls_id == 0:
                if bh < (h_orig * 0.10):  # Fondo lejano
                    continue
                if bw / float(bh) > 1.3:   # Personas sentadas
                    continue
            
            if cls_id == 32:
                ball_centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                
            action_boxes.append((x1, y1, x2, y2))

    # Si no detecta nada relevante
    if not action_boxes:
        min_x, min_y, max_x, max_y = int(w_orig * 0.15), int(h_orig * 0.15), int(w_orig * 0.85), int(h_orig * 0.85)
    else:
        min_x = min(b[0] for b in action_boxes)
        min_y = min(b[1] for b in action_boxes)
        max_x = max(b[2] for b in action_boxes)
        max_y = max(b[3] for b in action_boxes)

    subject_w = max_x - min_x
    subject_h = max_y - min_y

    # 1. PADDING ADAPTATIVO: Si la persona está muy cerca, usar menos margen
    if subject_h > (h_orig * 0.55):
        pad_ratio = 0.08  # 8% de margen si ya es un plano cercano
    else:
        pad_ratio = 0.20  # 20% si la persona está más lejos

    pad_x = int(subject_w * pad_ratio)
    pad_y = int(subject_h * pad_ratio)
    
    action_w = subject_w + (pad_x * 2)
    action_h = subject_h + (pad_y * 2)

    # Definir aspecto deseado (4:5 por defecto)
    target_aspect = 1.0 if action_w > (action_h * 1.25) else 0.8

    if target_aspect == 0.8:
        crop_h = max(action_h, int(action_w / 0.8))
        crop_w = int(crop_h * 0.8)
    else:
        crop_w = max(action_w, action_h)
        crop_h = crop_w

    # Limitar dimensiones al tamaño real de la foto
    if crop_w > w_orig:
        crop_w = w_orig
        crop_h = int(crop_w / target_aspect)
    if crop_h > h_orig:
        crop_h = h_orig
        crop_w = int(crop_h * target_aspect)

    # 2. CONTROL DE POSICIÓN SAGRADO (Proteger la cabeza):
    # La parte superior del crop intenta arrancar justo arriba de la cabeza (min_y - pad_y)
    top = min_y - pad_y
    
    # Ajustes si el balón o la atracción de jugada mueven la caja
    if ball_centers:
        avg_ball_y = sum(b[1] for b in ball_centers) // len(ball_centers)
        if avg_ball_y < min_y: # Si el balón está por encima de la persona
            top = min(top, avg_ball_y - pad_y)

    # Garantizar que el cuadro no se salga por arriba ni por abajo
    top = max(0, top)
    if top + crop_h > h_orig:
        top = h_orig - crop_h

    # Centrado horizontal respetando bordes
    center_x = (min_x + max_x) // 2
    if ball_centers:
        avg_ball_x = sum(b[0] for b in ball_centers) // len(ball_centers)
        center_x = int(center_x * 0.75 + avg_ball_x * 0.25)

    left = max(0, min(w_orig - crop_w, center_x - (crop_w // 2)))
    bottom = top + crop_h
    right = left + crop_w

    cropped_img = pil_img.crop((left, top, right, bottom))
    
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}.jpg")
    cropped_img.save(output_path, "JPEG", quality=95)
    print(f"   ✂️ Recortada [{ '4:5' if target_aspect == 0.8 else '1:1' }]: {base_name}.jpg")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DISCARD_FOLDER, exist_ok=True)
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(EXTENSIONS) and not filename.lower().startswith("logo"):
            procesar_foto_crop(filename)
            
    print("\n¡Proceso finalizado!")