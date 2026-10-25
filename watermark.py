import os
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

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
LOGOS_TO_SKIP = {LOGO_BOTTOM_PATH.lower(), LOGO_TOP_PATH.lower()}

# Carga global de logos
logo_bottom_base = None
logo_top_base = None

def get_logos():
    global logo_bottom_base, logo_top_base
    if logo_bottom_base is None and os.path.exists(LOGO_BOTTOM_PATH):
        logo_bottom_base = Image.open(LOGO_BOTTOM_PATH).convert("RGBA")
    if logo_top_base is None and os.path.exists(LOGO_TOP_PATH):
        logo_top_base = Image.open(LOGO_TOP_PATH).convert("RGBA")
    return logo_bottom_base, logo_top_base

def procesar_foto_watermark(filename, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    """Procesa una sola foto para aplicar marcas de agua."""
    bot_logo, top_logo = get_logos()
    
    img_path = os.path.join(input_folder, filename)
    raw_img = Image.open(img_path)
    img = ImageOps.exif_transpose(raw_img).convert("RGBA")
    
    margin_x = int(img.width * MARGIN_RATIO)
    margin_y = int(img.height * MARGIN_RATIO)
    
    # 1. Logo Inferior
    if bot_logo:
        target_w = int(img.width * LOGO_BOT_SCALE)
        aspect = bot_logo.height / bot_logo.width
        target_h = int(target_w * aspect)
        logo_bot_resized = bot_logo.resize((target_w, target_h), Image.Resampling.LANCZOS)
        x_bot = (img.width - target_w) // 2
        y_bot = img.height - target_h - margin_y
        img.paste(logo_bot_resized, (x_bot, y_bot), logo_bot_resized)

    # 2. Logo Transparente Superior
    if top_logo:
        target_w = int(img.width * LOGO_TOP_SCALE)
        aspect = top_logo.height / top_logo.width
        target_h = int(target_w * aspect)
        logo_top_resized = top_logo.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        r, g, b, alpha = logo_top_resized.split()
        alpha = alpha.point(lambda p: int(p * TOP_OPACITY))
        logo_top_resized = Image.merge("RGBA", (r, g, b, alpha))
        
        x_top = margin_x
        y_top = margin_y
        img.paste(logo_top_resized, (x_top, y_top), logo_top_resized)

    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}.jpg")
    
    img.convert("RGB").save(
        output_path, 
        "JPEG", 
        quality=85,
        progressive=True,
        optimize=True
    )
    print(f"Marca de agua aplicada: {base_name}.jpg")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(EXTENSIONS) and filename.lower() not in LOGOS_TO_SKIP:
            procesar_foto_watermark(filename)

    print("¡Proceso completado!")