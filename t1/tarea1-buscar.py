# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

import sys
import os
import util as util
import gc
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

def tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados):
    if not os.path.isdir(dir_input_imagenes_Q):
        print("ERROR: no existe directorio {}".format(dir_input_imagenes_Q))
        sys.exit(1)
    elif not os.path.isdir(dir_input_descriptores_R):
        print("ERROR: no existe directorio {}".format(dir_input_descriptores_R))
        sys.exit(1)
    elif os.path.exists(file_output_resultados):
        print("ERROR: ya existe archivo {}".format(file_output_resultados))
        sys.exit(1)
    # Implementar la fase online

    # 1-calcular descriptores de Q para imágenes en dir_input_imagenes_Q
    # ver codigo de ejemplo publicado en el curso
    print("1-calcular descriptores de Q para imágenes en dir_input_imagenes_Q")
    imagenes_q = util.listar_archivos_en_carpeta(dir_input_imagenes_Q)

    # Usar ThreadPoolExecutor para procesar las imágenes en paralelo
    batch_size = 10  

    descriptores_q = {}

    with ProcessPoolExecutor() as executor:
        # Dividir imágenes en lotes y procesar cada lote en paralelo
        batches = [imagenes_q[i:i+batch_size] for i in range(0, len(imagenes_q), batch_size)]
        futures = [executor.submit(util.procesar_imagenes_batch, batch, dir_input_imagenes_Q) for batch in batches]
        
        for future in as_completed(futures):
            batch_descriptores = future.result()
            descriptores_q.update(batch_descriptores)


    # 2-leer descriptores de R guardados en dir_input_descriptores_R
    # puede servir la funcion util.leer_objeto() que está definida en util.py
    print("2-leer descriptores de R guardados en dir_input_descriptores_R")
    descriptores_r = util.leer_objeto(dir_input_descriptores_R, 'descriptores.pkl')

    # 3-para cada descriptor q localizar el mas cercano en R
    print("3-para cada descriptor q localizar el mas cercano en R")
    tipos_descriptor = ['grayscale', 'fft', 'color']
    nombres_imagenes_r, descriptores_arrays_r = util.preparar_descriptores(descriptores_r, tipos_descriptor)
    nombres_imagenes_q, descriptores_arrays_q = util.preparar_descriptores(descriptores_q, tipos_descriptor)


    batch_size = 10  # Ajustar según la RAM disponible
    imagenes_q_procesadas = set()  # Para almacenar las imágenes Q ya procesadas

    # Usar ProcessPoolExecutor para paralelizar las consultas
    with ProcessPoolExecutor() as executor:
        for tipo_descriptor in ['grayscale', 'fft', 'color']:
            print(tipo_descriptor)
            resultados = []

            for idx_q in range(0, len(nombres_imagenes_q), batch_size):
                print(idx_q, len(nombres_imagenes_q))
                batch_descriptores_q = descriptores_arrays_q[tipo_descriptor][idx_q:idx_q + batch_size]
                
                for i, imagen_q in enumerate(nombres_imagenes_q[idx_q:idx_q + batch_size]):
                    
                    # Si la imagen Q ya fue procesada, saltar
                    if imagen_q in imagenes_q_procesadas:
                        continue
                    
                    mejores_resultados = {'imagen_r': None, 'distancia': float('inf')}
                    
                    # Procesar lotes de referencias en paralelo
                    futuros = [
                        executor.submit(util.calcular_distancias_min, batch_descriptores_q[i], descriptores_arrays_r[tipo_descriptor][idx_r:idx_r + batch_size])
                        for idx_r in range(0, len(nombres_imagenes_r), batch_size)
                    ]

                    for futuro in futuros:
                        min_dist, min_idx = futuro.result()
                        imagen_r = nombres_imagenes_r[min_idx]

                        if min_dist < mejores_resultados['distancia']:
                            mejores_resultados = {
                                'imagen_r': imagen_r,
                                'distancia': min_dist
                            }

                    # Guardar el mejor resultado para esta imagen de consulta
                    resultados.append([
                        imagen_q,
                        mejores_resultados['imagen_r'],
                        tipo_descriptor,
                        mejores_resultados['distancia']
                    ])

                    # Marcar la imagen Q como procesada
                    imagenes_q_procesadas.add(imagen_q)

            # Guardar resultados de este tipo_descriptor en un archivo
            print("Guardar resultados de este tipo_descriptor en un archivo")
            util.escribir_lista_de_columnas_en_archivo(resultados, file_output_resultados)
            print(f"Resultados para {tipo_descriptor} guardados en {file_output_resultados}")
            
            # Eliminar los resultados de memoria y forzar la recolección de basura
            del resultados
            gc.collect()

    # 4-escribir en el archivo file_output_resultados un archivo con tres columnas separado por \t:
    # columna 1: imagen_q
    # columna 2: imagen_r
    # columna 3: distancia
    # Puede servir la funcion util.escribir_lista_de_columnas_en_archivo() que está definida util.py
    #print("4-escribir en el archivo file_output_resultados un archivo con tres columnas separado por \t:")
    #util.escribir_lista_de_columnas_en_archivo(resultados, file_output_resultados)
    print(f"Resultados guardados en {file_output_resultados}")



# inicio de la tarea
if len(sys.argv) < 4:
    print("Uso: {} [dir_input_imagenes_Q] [dir_input_descriptores_R] [file_output_resultados]".format(sys.argv[0]))
    sys.exit(1)

# lee los parametros de entrada
dir_input_imagenes_Q = sys.argv[1]
dir_input_descriptores_R = sys.argv[2]
file_output_resultados = sys.argv[3]

# ejecuta la tarea
tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados)
