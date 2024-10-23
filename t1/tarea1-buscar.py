# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

import sys
import os
import util as util
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
        descriptor_q_grayscale = util.calcular_descriptores_grayscale(image)
        descriptor_flip_h, descriptor_flip_v, descriptor_flip_both  = util.calcular_descriptores_flip(image)
        #descriptor_q_color = util.calcular_histograma_color(image)
        #descriptor_gaussiano = util.calcular_descriptor_gaussiano(image)
        #descriptor_hsv = util.calcular_histograma_hsv(image)
        #descriptor_hog = util.calcular_descriptores_hog(image)
        #descriptor_hsv_1, descriptor_hsv_2, descriptor_hsv_3 = util.calcular_histograma_hsv_normalizado(image)
        
        descriptores_q[imagen_nombre] = {
            'grayscale': descriptor_q_grayscale,
            #'color': descriptor_q_color,
            'flip_h': descriptor_flip_h,
            'flip_v': descriptor_flip_v,
            'flip_both': descriptor_flip_both,
            #'gaussian': descriptor_gaussiano,
            #'hsv': descriptor_hsv,
        }
    # 2-leer descriptores de R guardados en dir_input_descriptores_R
    print("2-leer descriptores de R guardados en dir_input_descriptores_R")
    # puede servir la funcion util.leer_objeto() que está definida en util.py
    descriptores_r = util.leer_objeto(dir_input_descriptores_R, 'descriptores.pkl')

    # 3-para cada descriptor q localizar el más cercano en R con progreso
    print("3-para cada descriptor q localizar el más cercano en R con progreso")
    
    resultados = []

    descriptores_r_matriz_grayscale = np.array([descriptor_r['grayscale'] for descriptor_r in descriptores_r.values()])
    #descriptores_r_matriz_color = np.array([descriptor_r['color'] for descriptor_r in descriptores_r.values()])
    #descriptores_r_matriz_gaussiano = np.array([descriptor_r['gaussian'] for descriptor_r in descriptores_r.values()])
    #descriptors_r_matriz_hsv = np.array([descriptor_r['hsv'] for descriptor_r in descriptores_r.values()])
    #descriptores_r_matriz_hog= np.array([descriptor_r['hog'] for descriptor_r in descriptores_r.values()])
    #descriptores_r_matriz_hsv_1= np.array([descriptor_r['hsv_1'] for descriptor_r in descriptores_r.values()])
    #descriptores_r_matriz_hsv_2= np.array([descriptor_r['hsv_2'] for descriptor_r in descriptores_r.values()])
    #descriptores_r_matriz_hsv_3= np.array([descriptor_r['hsv_3'] for descriptor_r in descriptores_r.values()])
    imagenes_r = list(descriptores_r.keys())

    for imagen_q, descriptor_q in tqdm(descriptores_q.items(), desc="Procesando descriptores de Q"):
        distancia_minima = float('inf')
        imagen_r_minima = None

        #umbral_hog = 0.3
        #umbral_hog = np.inf
        distancias_grayscale = np.linalg.norm(descriptores_r_matriz_grayscale - descriptor_q['grayscale'], axis=1)
        #distancias_color = np.linalg.norm(descriptores_r_matriz_color - descriptor_q['color'], axis=1)
        distancias_flip_h = np.linalg.norm(descriptores_r_matriz_grayscale - descriptor_q['flip_h'], axis=1)
        distancias_flip_v = np.linalg.norm(descriptores_r_matriz_grayscale - descriptor_q['flip_v'], axis=1)
        distancias_flip_both = np.linalg.norm(descriptores_r_matriz_grayscale - descriptor_q['flip_both'], axis=1)
        #distancias_gauss = np.linalg.norm(descriptores_r_matriz_gaussiano - descriptor_q['gaussian'], axis=1)
        #distancias_hog = np.linalg.norm(descriptors_r_matriz_hsv - descriptor_q['hsv'], axis=1)

        # Aplicar umbral a distancias HOG
        #distancias_hog_ajustada = np.where(distancias_hog < umbral_hog, distancias_hog, np.inf)

        # Buscar la distancia mínima y su índice
        #distancias_totales = np.minimum.reduce([distancias_hog, distancias_grayscale, distancias_color, distancias_gauss, distancias_flip_h, distancias_flip_v, distancias_flip_both])

        #distancias_totales = np.minimum.reduce([distancias_grayscale])
        distancias_totales = np.minimum.reduce([distancias_grayscale, distancias_flip_h, distancias_flip_v, distancias_flip_both])
        indice_minimo = np.argmin(distancias_totales)

        imagen_r_minima = imagenes_r[indice_minimo]
        distancia_minima = distancias_totales[indice_minimo]

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
