# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 09 de septiembre de 2024
# Alumno: Ignacio Albornoz Alfaro

# este archivo es usado por tarea1-buscar.py y tarea1-buscar.py
# permite tener funciones compartidas entre ambos programas
import os
import pickle

import cv2
import numpy as np

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



def calcular_descriptores_grayscale(image):
    """Calcula descriptores en escala de grises para una imagen dada."""
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram in grayscale
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    return histogram.flatten()


def calcular_descriptores_fft(image):
    """Calcula la Transformada de Fourier (DFT) para una imagen."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # Escalar el espectro
    return magnitude_spectrum.flatten()


def calcular_histograma_color(image):
    """Calcula el histograma de colores de una imagen."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def calcular_distancia(descriptor_q, descriptor_r):
    """Calcula la distancia entre dos descriptores usando la distancia euclidiana."""
    return np.linalg.norm(descriptor_q - descriptor_r)
