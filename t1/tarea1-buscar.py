# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

import sys
import os
import util as util
import itertools
import numpy as np
import cv2
from tqdm import tqdm
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
    print("1-calcular descriptores de Q para imágenes en dir_input_imagenes_Q")
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
        descriptor_q_grayscale, grayscale_image = util.calcular_descriptores_grayscale(image)
        descriptor_q_bordes = util.calcular_descriptor_bordes(grayscale_image)
        descriptor_q_color = util.calcular_histograma_color(image)
        
        descriptores_q[imagen_nombre] = {
            'grayscale': descriptor_q_grayscale,
            'bordes': descriptor_q_bordes,
            'color': descriptor_q_color
        }
    # 2-leer descriptores de R guardados en dir_input_descriptores_R
    print("2-leer descriptores de R guardados en dir_input_descriptores_R")
    # puede servir la funcion util.leer_objeto() que está definida en util.py
    descriptores_r = util.leer_objeto(dir_input_descriptores_R, 'descriptores.pkl')

    # 3-para cada descriptor q localizar el más cercano en R con progreso
    print("3-para cada descriptor q localizar el más cercano en R con progreso")
    
    resultados = []
    
    # Iterar sobre cada imagen Q con tqdm para mostrar el progreso
    for imagen_q, descriptor_q in tqdm(descriptores_q.items(), desc="Procesando descriptores de Q"):
        # Para cada imagen Q inicializamos una distancia mínima infinita
        distancia_minima = float('inf')
        imagen_r_minima = None
        #combinacion_usada = None

        # Comparar con cada descriptor de R y mostrar el progreso para cada comparación
        for imagen_r, descriptor_r in tqdm(descriptores_r.items(), desc=f"Comparando {imagen_q} con descriptores de R", leave=False):
            
            # Comparar usando cada descriptor individualmente
            distancias = []

            distancia_grayscale = util.calcular_distancia(descriptor_q['grayscale'], descriptor_r['grayscale'])
            distancias.append(('grayscale', distancia_grayscale))

            distancia_bordes = util.calcular_distancia(descriptor_q['bordes'], descriptor_r['bordes'])
            distancias.append(('bordes', distancia_bordes))
                
            distancia_color = util.calcular_distancia(descriptor_q['color'], descriptor_r['color'])
            distancias.append(('color', distancia_color))

            # Comparar usando combinaciones de descriptores (grayscale+fft, grayscale+color, fft+color, grayscale+fft+color)
            if len(distancias) > 1:
                for i in range(2, len(distancias)+1):
                    for combinacion in itertools.combinations(distancias, i):
                        combinacion_distancia = sum([d[1] for d in combinacion])
                        combinacion_nombre = '+'.join([d[0] for d in combinacion])
                        distancias.append((combinacion_nombre, combinacion_distancia))

            # Buscar la distancia mínima entre todas las distancias (individuales y combinadas)
            for _, distancia in distancias:
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    imagen_r_minima = imagen_r
                    #combinacion_usada = descriptor_combination
        
        # Guardar el resultado (imagen_q, imagen_r_minima, distancia_minima, combinación usada)
        resultados.append([imagen_q, imagen_r_minima, distancia_minima])


    # 4-escribir en el archivo file_output_resultados un archivo con tres columnas separado por \t:
    print(f"4-escribir resultados en {file_output_resultados}")
    
    # Escribir los resultados en el archivo
    util.escribir_lista_de_columnas_en_archivo(resultados, file_output_resultados)
    
    print(f"Resultados guardados en {file_output_resultados}")
    # columna 1: imagen_q
    # columna 2: imagen_r
    # columna 3: distancia
    # Puede servir la funcion util.escribir_lista_de_columnas_en_archivo() que está definida util.py
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
