# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

# este archivo es usado por tarea1-buscar.py y tarea1-buscar.py
# permite tener funciones compartidas entre ambos programas
import os
import pickle

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Retorna tods los archivos .jpg que estan en una carpeta
def listar_archivos_en_carpeta(imagenes_dir):
    lista = []
    for archivo in os.listdir(imagenes_dir):
        # los que terminan en .jpg se agregan a la lista de nombres
        if archivo.endswith(".jpg"):
            lista.append(archivo)
    lista.sort()
    return lista


# escribe el objeto de python en un archivo binario
def guardar_objeto(objeto, carpeta, nombre_archivo):
    # asegura que la carpeta exista
    os.makedirs(carpeta, exist_ok=True)
    # nombre completo
    archivo = "{}/{}".format(carpeta, nombre_archivo)
    # usa la librería pickle para escribir el objeto en un archivo binario
    # ver https://docs.python.org/3/library/pickle.html
    with open(archivo, 'wb') as handle:
        pickle.dump(objeto, handle, protocol=pickle.HIGHEST_PROTOCOL)


# reconstruye el objeto de python que está guardado en un archivo
def leer_objeto(carpeta, nombre_archivo):
    archivo = "{}/{}".format(carpeta, nombre_archivo)
    with open(archivo, 'rb') as handle:
        objeto = pickle.load(handle)
    return objeto


# Recibe una lista de listas y lo escribe en un archivo separado por \t
# Por ejemplo:
# listas = [
#           ["dato1a", "dato1b", "dato1c"],
#           ["dato2a", "dato2b", "dato2c"],
#           ["dato3a", "dato3b", "dato3c"] ]
# al llamar:
#   escribir_lista_de_columnas_en_archivo(listas, "archivo.txt")
# escribe un archivo de texto con:
# dato1a  dato1b   dato1c
# dato2a  dato2b   dato3c
# dato2a  dato2b   dato3c
def escribir_lista_de_columnas_en_archivo(lista_con_columnas, archivo_texto_salida):
    with open(archivo_texto_salida, 'w') as handle:
        for columnas in lista_con_columnas:
            textos = []
            for col in columnas:
                textos.append(str(col))
            texto = "\t".join(textos)
            print(texto, file=handle)
            

def dividir_en_zonas(image, num_zonas=4):
    """Divide una imagen en zonas de num_zonas x num_zonas."""
    height, width = image.shape[:2]
    zona_h = height // num_zonas
    zona_w = width // num_zonas
    zonas = np.array([image[i*zona_h:(i+1)*zona_h, j*zona_w:(j+1)*zona_w]
                      for i in range(num_zonas) for j in range(num_zonas)])
    return zonas

def calcular_descriptores_grayscale(image):
    """Calcula y normaliza descriptores en escala de grises para cada zona de una imagen."""
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zonas = dividir_en_zonas(grayscale_image)
    descriptores = []

    for zona in zonas:
        # Histograma en escala de grises con 64 bins
        histograma = cv2.calcHist([zona], [0], None, [64], [0, 256])
        
        # Normalizar el histograma
        histograma = cv2.normalize(histograma, histograma).flatten()
        descriptores.extend(histograma)

    return np.array(descriptores), grayscale_image


def calcular_histograma_color(image):
    """Calcula y normaliza el histograma de colores para cada zona de una imagen."""
    zonas = dividir_en_zonas(image)
    descriptores = []

    for zona in zonas:
        hist = cv2.calcHist([zona], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        
        # Normalizar el histograma
        hist = cv2.normalize(hist, hist).flatten()
        descriptores.extend(hist)

    return np.array(descriptores)



def calcular_descriptores_orb(image):
    """Calcula descriptores por zonas para las versiones reflejadas horizontal, vertical y ambas direcciones."""
    
    # Flip horizontal
    flipped_horizontal = cv2.flip(image, 1)
    # Flip vertical
    flipped_vertical = cv2.flip(image, 0)
    # Flip en ambas direcciones (horizontal y vertical)
    flipped_both = cv2.flip(image, -1)
    
    # Inicializar los descriptores
    descriptores_flip_h = []
    descriptores_flip_v = []
    descriptores_flip_both = []
    
    # Dividir en zonas y calcular los descriptores para cada versión reflejada
    zonas_horizontal = dividir_en_zonas(flipped_horizontal)
    zonas_vertical = dividir_en_zonas(flipped_vertical)
    zonas_both = dividir_en_zonas(flipped_both)
    
    for zona in zonas_horizontal:
        hist_h = cv2.calcHist([zona], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        descriptores_flip_h.extend(hist_h)

    for zona in zonas_vertical:
        hist_v = cv2.calcHist([zona], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        descriptores_flip_v.extend(hist_v)

    for zona in zonas_both:
        hist_both = cv2.calcHist([zona], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        hist_both = cv2.normalize(hist_both, hist_both).flatten()
        descriptores_flip_both.extend(hist_both)
    
    return np.array(descriptores_flip_h), np.array(descriptores_flip_v), np.array(descriptores_flip_both)


def calcular_descriptor_gaussiano(image):
    """Aplica un filtro gaussiano por zonas y calcula descriptores normalizados por zonas."""
    # Convertir a escala de grises
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro gaussiano
    blurred_image = cv2.GaussianBlur(grayscale_image, (15, 15), 0)
    
    # Dividir en zonas
    zonas = dividir_en_zonas(blurred_image)
    descriptores = []

    # Calcular descriptores por zonas
    for zona in zonas:
        histograma = cv2.calcHist([zona], [0], None, [64], [0, 256])
        histograma = cv2.normalize(histograma, histograma).flatten()
        descriptores.extend(histograma)
    
    return np.array(descriptores)

