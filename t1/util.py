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

def calcular_distancias_min(descriptor_q, batch_descriptores_r):
    # Encontrar el índice de la mínima distancia sin almacenar todas las distancias
    min_distancia = float('inf')
    min_idx = -1
    for i in range(batch_descriptores_r.shape[0]):
        distancia = np.linalg.norm(batch_descriptores_r[i] - descriptor_q)
        if distancia < min_distancia:
            min_distancia = distancia
            min_idx = i
    return min_distancia, min_idx


def preparar_descriptores(descriptores, tipos_descriptor):
    print(" Preparando los descriptores...")
    nombres_imagenes = list(descriptores.keys())
    descriptores_arrays = {tipo: [] for tipo in tipos_descriptor}
    
    for imagen in nombres_imagenes:
        for tipo in tipos_descriptor:
            descriptores_arrays[tipo].append(descriptores[imagen][tipo])
    
    # Convertir listas a arrays de NumPy una vez para cada descriptor
    for tipo in tipos_descriptor:
        descriptores_arrays[tipo] = np.array(descriptores_arrays[tipo], dtype=np.float32)  # Usa float32 para reducir memoria
    
    return nombres_imagenes, descriptores_arrays
'''
def preparar_descriptores(descriptores, tipos_descriptor):
    nombres_imagenes = list(descriptores.keys())
    descriptores_arrays = {tipo: [] for tipo in tipos_descriptor}
    
    for imagen in nombres_imagenes:
        for tipo in tipos_descriptor:
            descriptores_arrays[tipo].append(descriptores[imagen][tipo])
    
    # Convertir listas a arrays de NumPy una vez para cada descriptor
    for tipo in tipos_descriptor:
        descriptores_arrays[tipo] = np.array(descriptores_arrays[tipo], dtype=np.float32)  # Usa float32 para reducir memoria
    
    return nombres_imagenes, descriptores_arrays

def calcular_distancias_batch(descriptor_q, batch_descriptores_r):
    # Calcula la distancia para todo el lote de una sola vez
    distancias = np.linalg.norm(batch_descriptores_r - descriptor_q, axis=1)
    return distancias
'''

def calcular_descriptores_grayscale(images):
    """Calcula descriptores en escala de grises para un lote de imágenes."""
    # Convertir todas las imágenes a escala de grises en un solo paso
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    
    # Calcular el histograma de todas las imágenes en un solo paso
    histograms = [cv2.calcHist([grayscale_image], [0], None, [256], [0, 256]).flatten() for grayscale_image in grayscale_images]
    
    return np.array(histograms)


def calcular_descriptores_fft(images):
    """Calcula la Transformada de Fourier (DFT) para un lote de imágenes."""
    # Convertir todas las imágenes a escala de grises en un solo paso
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    
    # Calcular la FFT de todas las imágenes y devolver la magnitud del espectro
    magnitude_spectrums = [20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))) + 1).flatten() for gray_image in gray_images]
    
    return np.array(magnitude_spectrums)


def calcular_histograma_color(images):
    """Calcula el histograma de colores de un lote de imágenes."""
    histograms = [cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten() for image in images]
    
    # Normalización
    histograms = [cv2.normalize(hist, hist).flatten() for hist in histograms]
    
    return np.array(histograms)



def procesar_imagenes_batch(imagenes, dir_input_imagenes_R, target_size=(256, 256)):
    image_paths = [os.path.join(dir_input_imagenes_R, imagen_nombre) for imagen_nombre in imagenes]
    images = [cv2.imread(image_path) for image_path in image_paths]
    
    # Filtrar imágenes que no se hayan podido leer
    valid_images = [(image, imagen_nombre) for image, imagen_nombre in zip(images, imagenes) if image is not None]
    images, imagenes_validas = zip(*valid_images)
    
    # Redimensionar todas las imágenes a un tamaño común (256x256 o el que prefieras)
    resized_images = [cv2.resize(image, target_size) for image in images]
    
    # Calcular descriptores en batch
    grayscale_descriptors = calcular_descriptores_grayscale(resized_images)
    fft_descriptors = calcular_descriptores_fft(resized_images)
    color_histograms = calcular_histograma_color(resized_images)
    
    # Crear diccionario de resultados
    descriptores = {
        imagen_nombre: {
            'grayscale': grayscale.tolist(),
            'fft': fft.tolist(),
            'color': color_hist.tolist()
        }
        for imagen_nombre, grayscale, fft, color_hist in zip(imagenes_validas, grayscale_descriptors, fft_descriptors, color_histograms)
    }
    
    return descriptores

