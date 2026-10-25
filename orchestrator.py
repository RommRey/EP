import os
import argparse
from multiprocessing import Pool, cpu_count
import time

from cropper import procesar_foto_crop, init_worker as init_crop_worker, EXTENSIONS as CROP_EXTS
from watermark import procesar_foto_watermark, EXTENSIONS as WM_EXTS

def worker_crop(args):
    filename, input_folder, output_folder, discard_folder, threshold = args
    try:
        procesar_foto_crop(filename, input_folder, output_folder, discard_folder, threshold)
    except Exception as e:
        print(f"Error procesando {filename} en crop: {e}")

def worker_watermark(args):
    filename, input_folder, output_folder = args
    try:
        procesar_foto_watermark(filename, input_folder, output_folder)
    except Exception as e:
        print(f"Error procesando {filename} en watermark: {e}")

def run_multiprocessing(task, input_folder, output_folder, discard_folder, threshold, num_workers):
    print(f"\n🚀 Iniciando tarea '{task}' con {num_workers} núcleos en paralelo...")
    start_time = time.time()

    if task == "crop":
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(discard_folder, exist_ok=True)
        
        files = [
            f for f in os.listdir(input_folder) 
            if f.lower().endswith(CROP_EXTS) and not f.lower().startswith("logo")
        ]
        
        tasks_args = [
            (f, input_folder, output_folder, discard_folder, threshold) 
            for f in files
        ]

        # Usamos initializer para cargar YOLO una sola vez por trabajador
        with Pool(processes=num_workers, initializer=init_crop_worker) as pool:
            pool.map(worker_crop, tasks_args)

    elif task == "watermark":
        os.makedirs(output_folder, exist_ok=True)
        
        LOGOS_TO_SKIP = {"logo.png", "logo_transparente.png"}
        files = [
            f for f in os.listdir(input_folder) 
            if f.lower().endswith(WM_EXTS) and f.lower() not in LOGOS_TO_SKIP
        ]
        
        tasks_args = [
            (f, input_folder, output_folder) 
            for f in files
        ]

        with Pool(processes=num_workers) as pool:
            pool.map(worker_watermark, tasks_args)

    elapsed = time.time() - start_time
    print(f"✨ Tarea '{task}' completada en {elapsed:.2f} segundos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador Multiprocesamiento para Fotos")
    parser.add_argument("--task", choices=["crop", "watermark"], required=True, help="Tarea a ejecutar")
    parser.add_argument("--input", default=None, help="Carpeta de entrada")
    parser.add_argument("--output", default=None, help="Carpeta de salida")
    parser.add_argument("--discard", default="descartadas", help="Carpeta de descartadas (solo crop)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Umbral de nitidez")
    parser.add_argument("--workers", type=int, default=None, help="Número de núcleos a usar")

    args = parser.parse_args()

    # Ajuste por defecto inteligente según la tarea
    if args.task == "crop":
        if args.input is None:
            args.input = "naturales"
        if args.output is None:
            args.output = "recortes"
        if args.workers is None:
            args.workers = min(3, cpu_count())  # Límite ideal para YOLO para evitar saturar la CPU
    elif args.task == "watermark":
        if args.input is None:
            args.input = "filtro"
        if args.output is None:
            args.output = "salida"
        if args.workers is None:
            args.workers = cpu_count()  # Pillow es ligero y sí puede usar todos los núcleos

    run_multiprocessing(
        task=args.task,
        input_folder=args.input,
        output_folder=args.output,
        discard_folder=args.discard,
        threshold=args.threshold,
        num_workers=args.workers
    )