# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

import sys
import os
import util as util

import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed


def tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R):
    if not os.path.isdir(dir_input_imagenes_R):
        print("ERROR: no existe directorio {}".format(dir_input_imagenes_R))
        sys.exit(1)
    elif os.path.exists(dir_output_descriptores_R):
        print("ERROR: ya existe directorio {}".format(dir_output_descriptores_R))
        sys.exit(1)
    # Implementar la fase offline
    
    # 1-leer imágenes en dir_input_imagenes
    # puede servir la funcion util.listar_archivos_en_carpeta() que está definida en util.py
    print("1-leer imágenes en dir_input_imagenes")
    imagenes = util.listar_archivos_en_carpeta(dir_input_imagenes_R)

    # 2-calcular descriptores de imágenes
    # ver codigo de ejemplo publicado en el curso
    print("2-calcular descriptores de imágenes")
    '''
    descriptores = {}

    for imagen_nombre in imagenes:
        image_path = os.path.join(dir_input_imagenes_R, imagen_nombre)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"ERROR: no se pudo leer la imagen {imagen_nombre}")
            continue
        
        # Calcular descriptores
        descriptor_grayscale = util.calcular_descriptores_grayscale(image).astype(np.float16)
        descriptor_fft = util.calcular_descriptores_fft(image).astype(np.float16)
        descriptor_color = util.calcular_histograma_color(image).astype(np.float16)
        
        # Guardar descriptores en un diccionario
        descriptores[imagen_nombre] = {
            'grayscale': descriptor_grayscale.tolist(),
            'fft': descriptor_fft.tolist(),
            'color_histogram': descriptor_color.tolist()
        }

    '''
    # Usar ThreadPoolExecutor para procesar las imágenes en paralelo
    batch_size = 10  

    descriptores = {}

    with ProcessPoolExecutor() as executor:
        # Dividir imágenes en lotes y procesar cada lote en paralelo
        batches = [imagenes[i:i+batch_size] for i in range(0, len(imagenes), batch_size)]
        futures = [executor.submit(util.procesar_imagenes_batch, batch, dir_input_imagenes_R) for batch in batches]
        
        for future in as_completed(futures):
            batch_descriptores = future.result()
            descriptores.update(batch_descriptores)

    # 3-escribir en dir_output_descriptores_R los descriptores calculados en uno o más archivos
    # puede servir la funcion util.guardar_objeto() que está definida en util.py
    print("3-escribir en dir_output_descriptores_R los descriptores calculados en uno o más archivos")
    util.guardar_objeto(descriptores, dir_output_descriptores_R, 'descriptores.pkl')
    print(f"Descriptores guardados en {dir_output_descriptores_R}")

# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {} [dir_input_imagenes_R] [dir_output_descriptores_R]".format(sys.argv[0]))
    sys.exit(1)

# lee los parametros de entrada
dir_input_imagenes_R = sys.argv[1]
dir_output_descriptores_R = sys.argv[2]

# ejecuta la tarea
tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R)
