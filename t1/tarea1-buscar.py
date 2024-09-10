# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

import sys
import os
import util as util

import numpy as np
import cv2

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

    descriptores_q = {}
    
    for imagen_nombre in imagenes_q:
        image_path = os.path.join(dir_input_imagenes_Q, imagen_nombre)
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: no se pudo leer la imagen {imagen_nombre}")
            continue
        
        # Calcula los descriptores de cada imagen
        descriptor_q_grayscale = util.calcular_descriptores_grayscale(image)
        descriptor_q_fft = util.calcular_descriptores_fft(image)
        descriptor_q_color = util.calcular_histograma_color(image)
        
        descriptores_q[imagen_nombre] = {
            'grayscale': descriptor_q_grayscale,
            'fft': descriptor_q_fft,
            'color': descriptor_q_color
        }
    

    # 2-leer descriptores de R guardados en dir_input_descriptores_R
    # puede servir la funcion util.leer_objeto() que está definida en util.py
    print("2-leer descriptores de R guardados en dir_input_descriptores_R")
    descriptores_r = util.leer_objeto(dir_input_descriptores_R, 'descriptores.pkl')

    # 3-para cada descriptor q localizar el mas cercano en R
    print("3-para cada descriptor q localizar el mas cercano en R")
    resultados = []
    
    for imagen_q, descriptores_q_data in descriptores_q.items():
        mejores_resultados = {
            'grayscale': {'imagen_r': None, 'distancia': float('inf')},
            'fft': {'imagen_r': None, 'distancia': float('inf')},
            'color': {'imagen_r': None, 'distancia': float('inf')}
        }

        for imagen_r, descriptores_r_data in descriptores_r.items():
            # Itera sobre cada tipo de descriptor por separado
            for tipo_descriptor in descriptores_q_data:
                descriptor_q = descriptores_q_data[tipo_descriptor]
                descriptor_r = descriptores_r_data[tipo_descriptor]
                distancia = util.calcular_distancia(np.array(descriptor_q), np.array(descriptor_r))
                
                # Si la distancia es menor, se actualiza el mejor resultado para ese descriptor
                if distancia < mejores_resultados[tipo_descriptor]['distancia']:
                    mejores_resultados[tipo_descriptor]['imagen_r'] = imagen_r
                    mejores_resultados[tipo_descriptor]['distancia'] = distancia

        # Agrega los mejores resultados para cada tipo de descriptor
        for tipo_descriptor in mejores_resultados:
            resultados.append([
                imagen_q,
                mejores_resultados[tipo_descriptor]['imagen_r'],
                tipo_descriptor,
                mejores_resultados[tipo_descriptor]['distancia']
            ])

    # 4-escribir en el archivo file_output_resultados un archivo con tres columnas separado por \t:
    # columna 1: imagen_q
    # columna 2: imagen_r
    # columna 3: distancia
    # Puede servir la funcion util.escribir_lista_de_columnas_en_archivo() que está definida util.py
    util.escribir_lista_de_columnas_en_archivo(resultados, file_output_resultados)
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
