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
    '''
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
    '''
    descriptores_q = util.procesar_imagenes_en_paralelo(imagenes_q, dir_input_imagenes_Q)
    # 2-leer descriptores de R guardados en dir_input_descriptores_R
    # puede servir la funcion util.leer_objeto() que está definida en util.py
    descriptores_r = util.leer_objeto(os.path.join(dir_input_descriptores_R, 'descriptores.pkl'))

    # Se convierten los descriptores de R en matrices NumPy para hacer los cálculos de manera vectorizada
    descriptores_r_grayscale = np.array([descriptores_r[imagen]['grayscale'] for imagen in descriptores_r])
    descriptores_r_fft = np.array([descriptores_r[imagen]['fft'] for imagen in descriptores_r])
    descriptores_r_color = np.array([descriptores_r[imagen]['color'] for imagen in descriptores_r])

    imagenes_r = list(descriptores_r.keys())  # Lista de nombres de las imágenes R

    # 3-para cada descriptor q localizar el mas cercano en R
    resultados = []
    
    for imagen_q, descriptores_q_data in descriptores_q.items():
        mejores_resultados = {
            'grayscale': {'imagen_r': None, 'distancia': float('inf')},
            'fft': {'imagen_r': None, 'distancia': float('inf')},
            'color': {'imagen_r': None, 'distancia': float('inf')}
        }

        # Calcula distancias para todos los descriptores de una sola vez
        for tipo_descriptor in descriptores_q_data:
            descriptor_q = descriptores_q_data[tipo_descriptor]
            
            if tipo_descriptor == 'grayscale':
                distancias = util.calcular_distancia_lote(descriptor_q, descriptores_r_grayscale)
            elif tipo_descriptor == 'fft':
                distancias = util.calcular_distancia_lote(descriptor_q, descriptores_r_fft)
            elif tipo_descriptor == 'color':
                distancias = util.calcular_distancia_lote(descriptor_q, descriptores_r_color)

            # Encuentra la distancia mínima y la imagen correspondiente
            indice_mejor = np.argmin(distancias)
            distancia_mejor = distancias[indice_mejor]
            imagen_r_mejor = imagenes_r[indice_mejor]

            mejores_resultados[tipo_descriptor]['imagen_r'] = imagen_r_mejor
            mejores_resultados[tipo_descriptor]['distancia'] = distancia_mejor

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
