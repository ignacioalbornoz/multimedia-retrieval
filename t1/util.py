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



def calcular_descriptores_flip(image):
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



from skimage.feature import hog

def calcular_histograma_hsv(image, num_zonas=4, size=(128, 128)):
    """Calcula el descriptor HOG por zonas, dividiendo la imagen en num_zonas x num_zonas zonas."""

    # Redimensionar la imagen a un tamaño estándar
    resized_image = cv2.resize(image, size)
    
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Dividir en zonas
    zonas = dividir_en_zonas(gray_image, num_zonas=num_zonas)
    descriptores = []
    
    # Calcular HOG por cada zona
    for zona in zonas:
        hog_descriptor = hog(zona, orientations=6, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        hog_descriptor = np.array(hog_descriptor, dtype=np.float32)
        descriptores.extend(hog_descriptor)
    
    return np.array(descriptores)


'''

from skimage.feature import hog
def calcular_histograma_hsv(image, size=(128, 128)):
    """Calcula el descriptor HOG para la imagen completa con optimización, redimensionando a un tamaño estándar."""
    # Redimensionar la imagen a un tamaño estándar
    resized_image = cv2.resize(image, size)
    
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Calcular HOG con bloques más grandes para reducir el número de cálculos
    hog_descriptor = hog(gray_image, orientations=6, pixels_per_cell=(16, 16),
                         cells_per_block=(4, 4), block_norm='L2-Hys', transform_sqrt=True)
    
    # Convertir a formato float32 para normalización
    hog_descriptor = np.array(hog_descriptor, dtype=np.float32)
    
    # Normalizar el descriptor
    hog_descriptor = cv2.normalize(hog_descriptor, hog_descriptor).flatten()

    return np.array(hog_descriptor, dtype=np.float32)

'''





'''

def calcular_histograma_hsv(image):
    """Calcula el descriptor HOG para la imagen completa."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular HOG para la imagen completa
    hog_descriptor = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    
    # Convertir a formato float32 para normalización
    hog_descriptor = np.array(hog_descriptor, dtype=np.float32)
    
    # Normalizar el descriptor
    hog_descriptor = cv2.normalize(hog_descriptor, hog_descriptor).flatten()

    return hog_descriptor


def calcular_histograma_hsv(image):
    """Calcula el descriptor HOG para cada zona de una imagen."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zonas = dividir_en_zonas(gray_image)
    descriptores = []

    for zona in zonas:
        # Calcular HOG para cada zona
        hog_descriptor = hog(zona, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        
        # Convertir a formato float32 para normalización
        hog_descriptor = np.array(hog_descriptor, dtype=np.float32)
        
        # Normalizar el descriptor
        hog_descriptor = cv2.normalize(hog_descriptor, hog_descriptor).flatten()
        
        descriptores.extend(hog_descriptor)

    return np.array(descriptores)


def calcular_descriptores_hog(image):
    """Calcula descriptores HOG en la parte de la imagen sin texto añadido."""
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crear el descriptor HOG
    hog = cv2.HOGDescriptor()
    hog_descriptor = hog.compute(grayscale_image)
    
    return np.array(hog_descriptor.flatten())


def calcular_descriptores_hog(image, num_zonas=4):
    """Calcula y normaliza descriptores HOG por zonas."""
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zonas = dividir_en_zonas(grayscale_image, num_zonas)
    descriptores = []
    
    # Crear el descriptor HOG
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),  # Ajusta este tamaño según tu imagen
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    
    for zona in zonas:
        # Asegurar que la zona tenga el tamaño correcto para HOG
        resized_zona = cv2.resize(zona, (64, 64))
        
        # Calcular el descriptor HOG
        hog_descriptor = hog.compute(resized_zona)
        descriptores.extend(hog_descriptor.flatten())
    
    return np.array(descriptores)


'''









'''


from skimage.feature import local_binary_pattern

def calcular_lbp(image, radius=1, n_points=8):
    """Calcula el descriptor LBP (Local Binary Patterns) para cada zona de una imagen."""
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Dividir la imagen en zonas
    zonas = dividir_en_zonas(gray_image)
    descriptores = []

    # Calcular LBP para cada zona
    for zona in zonas:
        lbp = local_binary_pattern(zona, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        
        # Normalizar el histograma
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Añadir epsilon para evitar división por cero
        descriptores.extend(hist)

    return np.array(descriptores)

    
from skimage.feature import local_binary_pattern

def calcular_histograma_lbp(image, radius=1, n_points=8):
    """Calcula el descriptor LBP (Local Binary Patterns) para cada zona de una imagen."""
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Dividir la imagen en zonas
    zonas = dividir_en_zonas(gray_image)
    descriptores = []

    # Calcular LBP para cada zona
    for zona in zonas:
        lbp = local_binary_pattern(zona, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        
        # Convertir el histograma a float32 para usar cv2.normalize
        hist = hist.astype(np.float32)
        
        # Normalizar el histograma utilizando cv2.normalize
        hist = cv2.normalize(hist, hist).flatten()
        descriptores.extend(hist)

    return np.array(descriptores)
'''