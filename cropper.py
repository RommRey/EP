import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO

# Registrar soporte para HEIC/HEIF
register_heif_opener()

INPUT_FOLDER = "naturales"
OUTPUT_FOLDER = "recortes"
DISCARD_FOLDER = "descartadas"


# Ajuste de nitidez
SHARPNESS_THRESHOLD_PERCENT = 10.0
REFERENCE_LAPLACIAN_VAR = 100.0  

model = YOLO("yolov8n.pt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DISCARD_FOLDER, exist_ok=True)

TARGET_CLASSES = [0, 32]
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.heic', '.heif')

def calcular_porcentaje_nitidez(pil_img):
    open_cv_image = np.array(pil_img.convert('L'))
    laplacian_var = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    percentage = (laplacian_var / REFERENCE_LAPLACIAN_VAR) * 100.0
    return min(percentage, 100.0), laplacian_var

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(EXTENSIONS) and not filename.lower().startswith("logo"):
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        raw_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
        
        nitidez_pct, lap_var = calcular_porcentaje_nitidez(pil_img)
        
        if nitidez_pct < SHARPNESS_THRESHOLD_PERCENT:
            discard_path = os.path.join(DISCARD_FOLDER, filename)
            shutil.copy(img_path, discard_path)
            print(f"❌ Descartada (Nitidez: {nitidez_pct:.1f}%): {filename}")
            continue

        print(f"✅ Pasa filtro (Nitidez: {nitidez_pct:.1f}%): Recortando {filename}...")
        w_orig, h_orig = pil_img.size

        results = model(pil_img, verbose=False)[0]
        action_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = x2 - x1, y2 - y1
                
                if cls_id == 0:
                    if bh < (h_orig * 0.12):
                        continue
                    if bw / float(bh) > 1.2:
                        continue
                
                action_boxes.append((x1, y1, x2, y2))

        if not action_boxes:
            min_x, min_y, max_x, max_y = int(w_orig * 0.2), int(h_orig * 0.2), int(w_orig * 0.8), int(h_orig * 0.8)
        else:
            min_x = min(b[0] for b in action_boxes)
            min_y = min(b[1] for b in action_boxes)
            max_x = max(b[2] for b in action_boxes)
            max_y = max(b[3] for b in action_boxes)

        pad_x = int((max_x - min_x) * 0.15)
        pad_y = int((max_y - min_y) * 0.15)
        
        action_w = (max_x - min_x) + (pad_x * 2)
        action_h = (max_y - min_y) + (pad_y * 2)
        
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # ------------------------------------------------------------------------------
        # PREFERENCIA 4:5 VERTICAL:
        # Solo usará 1:1 si la acción es súper ancha (ancho > 1.25 * alto).
        # En caso contrario, forzará siempre el encuadre 4:5 (ratio 0.8).
        # ------------------------------------------------------------------------------
        if action_w > (action_h * 1.25):
            target_aspect = 1.0  # Cuadrado 1:1 solo para jugadas muy extendidas horizontalmente
        else:
            target_aspect = 0.8  # 4:5 Vertical por preferencia predeterminada

        if target_aspect == 0.8:
            # Escalamos la altura o el ancho en función de mantener 4:5 abarcando todo
            crop_h = max(action_h, int(action_w / 0.8))
            crop_w = int(crop_h * 0.8)
        else:
            crop_w = max(action_w, action_h)
            crop_h = crop_w

        # Limitar para no salir de los bordes de la foto original
        if crop_w > w_orig:
            crop_w = w_orig
            crop_h = int(crop_w / target_aspect)
        if crop_h > h_orig:
            crop_h = h_orig
            crop_w = int(crop_h * target_aspect)

        left = max(0, min(w_orig - crop_w, center_x - (crop_w // 2)))
        top = max(0, min(h_orig - crop_h, center_y - (crop_h // 2)))
        right = left + crop_w
        bottom = top + crop_h

        cropped_img = pil_img.crop((left, top, right, bottom))
        
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.jpg")
        cropped_img.save(output_path, "JPEG", quality=95)
        print(f"   ✂️ Recortada [{ '4:5' if target_aspect == 0.8 else '1:1' }]: {base_name}.jpg")

print("\n¡Proceso finalizado!")