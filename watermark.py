import os
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Registrar soporte HEIC/HEIF
register_heif_opener()

INPUT_FOLDER = "filtro" if os.path.exists("filtro") else "."
OUTPUT_FOLDER = "salida"

LOGO_BOTTOM_PATH = "logo.png"
LOGO_TOP_PATH = "logo_transparente.png"

LOGO_BOT_SCALE = 90 / 1080
LOGO_TOP_SCALE = 225 / 1080
MARGIN_RATIO = 0.05
TOP_OPACITY = 0.25

EXTENSIONS = ('.jpg', '.jpeg', '.png', '.heic', '.heif')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logo_bottom_base = Image.open(LOGO_BOTTOM_PATH).convert("RGBA") if os.path.exists(LOGO_BOTTOM_PATH) else None
logo_top_base = Image.open(LOGO_TOP_PATH).convert("RGBA") if os.path.exists(LOGO_TOP_PATH) else None

LOGOS_TO_SKIP = {LOGO_BOTTOM_PATH.lower(), LOGO_TOP_PATH.lower()}

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(EXTENSIONS) and filename.lower() not in LOGOS_TO_SKIP:
        img_path = os.path.join(INPUT_FOLDER, filename)
        raw_img = Image.open(img_path)
        img = ImageOps.exif_transpose(raw_img).convert("RGBA")
        
        margin_x = int(img.width * MARGIN_RATIO)
        margin_y = int(img.height * MARGIN_RATIO)
        
        # 1. Logo Inferior
        if logo_bottom_base:
            target_w = int(img.width * LOGO_BOT_SCALE)
            aspect = logo_bottom_base.height / logo_bottom_base.width
            target_h = int(target_w * aspect)
            logo_bot_resized = logo_bottom_base.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x_bot = (img.width - target_w) // 2
            y_bot = img.height - target_h - margin_y
            img.paste(logo_bot_resized, (x_bot, y_bot), logo_bot_resized)

        # 2. Logo Transparent Superior
        if logo_top_base:
            target_w = int(img.width * LOGO_TOP_SCALE)
            aspect = logo_top_base.height / logo_top_base.width
            target_h = int(target_w * aspect)
            logo_top_resized = logo_top_base.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            r, g, b, alpha = logo_top_resized.split()
            alpha = alpha.point(lambda p: int(p * TOP_OPACITY))
            logo_top_resized = Image.merge("RGBA", (r, g, b, alpha))
            
            x_top = margin_x
            y_top = margin_y
            img.paste(logo_top_resized, (x_top, y_top), logo_top_resized)

        # Guardar en JPG optimizado para Instagram
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.jpg")
        
        img.convert("RGB").save(
            output_path, 
            "JPEG", 
            quality=85,
            progressive=True,
            optimize=True
        )
        print(f"Marca de agua aplicada: {base_name}.jpg")

print("¡Proceso completado!")