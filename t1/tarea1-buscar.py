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
    descriptores_r = util.leer_objeto(os.path.join(dir_input_descriptores_R, 'descriptores.pkl'))

    # 3-para cada descriptor q localizar el mas cercano en R
    resultados = []
    umbral = 0.1  # Umbral de cercanía para considerar una imagen como duplicada
    
    for imagen_q, descriptores_q_data in descriptores_q.items():
        duplicado_encontrado = False
        imagen_r_mas_parecida = None
        distancia_mas_cercana = float('inf')
        
        for imagen_r, descriptores_r_data in descriptores_r.items():
            # Itera sobre cada tipo de descriptor por separado
            for tipo_descriptor in descriptores_q_data:
                descriptor_q = descriptores_q_data[tipo_descriptor]
                descriptor_r = descriptores_r_data[tipo_descriptor]
                distancia = util.calcular_distancia(np.array(descriptor_q), np.array(descriptor_r))
                
                # Verifica si la distancia es menor al umbral
                if distancia < umbral:
                    duplicado_encontrado = True
                    if distancia < distancia_mas_cercana:
                        distancia_mas_cercana = distancia
                        imagen_r_mas_parecida = imagen_r
                    break  # Si ya es un duplicado, no es necesario verificar los demás descriptores
            
            if duplicado_encontrado:
                break

        # Agrega el resultado si se encontró un duplicado
        if duplicado_encontrado:
            resultados.append([imagen_q, imagen_r_mas_parecida, distancia_mas_cercana])


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
