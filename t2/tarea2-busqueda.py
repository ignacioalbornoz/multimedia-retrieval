# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [nombre]

import sys
import os
import util as util
import numpy as np
import scipy.spatial.distance as dist

def tarea2_busqueda(carpeta_descriptores_radio_Q, carpeta_descritores_canciones_R, archivo_ventanas_similares):
    if not os.path.isdir(carpeta_descriptores_radio_Q):
        print("ERROR: no existe {}".format(carpeta_descriptores_radio_Q))
        sys.exit(1)
    elif not os.path.isdir(carpeta_descritores_canciones_R):
        print("ERROR: no existe {}".format(carpeta_descritores_canciones_R))
        sys.exit(1)
    elif os.path.exists(archivo_ventanas_similares):
        print("ERROR: ya existe {}".format(archivo_ventanas_similares))
        sys.exit(1)
    #
    # Implementar la tarea con los siguientes pasos:
    #
    #  1-leer Q y R: datos en carpeta_descriptores_radio_Q y carpeta_descritores_canciones_R
    #     esas carpetas fueron creadas por tarea2_extractor con los audios de radio y canciones
    #     puede servir la funcion util.leer_objeto() que está definida en util.py
    # Constants for time calculations (these should match the values used in MFCC extraction)
    sample_rate = 22050
    hop_length = 512  # Number of samples between successive frames


    # Step 1: Load descriptors from Q (radio) and R (songs)
    descriptores_Q = []
    archivos_Q = util.listar_archivos_con_extension(carpeta_descriptores_radio_Q, ".pkl")
    for archivo_Q in archivos_Q:
        descriptores_Q.append(util.leer_objeto(carpeta_descriptores_radio_Q, archivo_Q))

    descriptores_R = []
    archivos_R = util.listar_archivos_con_extension(carpeta_descritores_canciones_R, ".pkl")
    lengths_R = []
    for archivo_R in archivos_R:
        descr_R = util.leer_objeto(carpeta_descritores_canciones_R, archivo_R)
        descriptores_R.append(descr_R)
        lengths_R.append(len(descr_R))

    #  2-para cada descriptor de Q localizar el más cercano en R
    #     podría usar cdist (ver semana 02) o algún índice de busqueda eficiente (Semanas 03-04)
    descriptores_R_flat = np.vstack(descriptores_R)
    batch_size = 5000
    resultados_similares = []
    # Calcular la duración de cada archivo R (antes del bucle principal)
    #duraciones_R = [(length * hop_length) / sample_rate for length in lengths_R]
    for i, desc_Q in enumerate(descriptores_Q):
        num_windows_Q = desc_Q.shape[0]

        for batch_start in range(0, num_windows_Q, batch_size):
            batch_end = min(batch_start + batch_size, num_windows_Q)
            batch_desc_Q = desc_Q[batch_start:batch_end]

            # Compute distances in manageable chunks
            distancias_batch = dist.cdist(batch_desc_Q, descriptores_R_flat, metric='euclidean')

            # Get the closest window in R for each window in the current batch of Q
            posiciones_min = np.argmin(distancias_batch, axis=1)
            minimas_distancias = np.min(distancias_batch, axis=1)

            # Find which file in R corresponds to the minimum position
            for j, pos_min in enumerate(posiciones_min):
                total_windows = 0
                archivo_R_idx = -1
                for idx, length in enumerate(lengths_R):
                    total_windows += length
                    if pos_min < total_windows:
                        archivo_R_idx = idx
                        break

                idx_ventana = pos_min - (total_windows - lengths_R[archivo_R_idx])

                # Calculate start and end times for the windows in Q and R
                start_time_Q = (batch_start + j) * hop_length / sample_rate
                start_time_R = idx_ventana * hop_length / sample_rate
                #duracion_R_en_ventanas = lengths_R[archivo_R_idx] 
                # Append the result to the list of similar windows
                resultados_similares.append([
                    archivos_Q[i], f"{start_time_Q:.2f}",  # Q file and start time
                    archivos_R[archivo_R_idx], f"{start_time_R:.2f}",  # R file and start time
                    f"{minimas_distancias[j]:.4f}" # Distance between the windows
                    #f"{duracion_R_en_ventanas:.2f}"  # Duración del archivo R
                ])
    #  3-escribir en el archivo archivo_ventanas_similares una estructura que asocie
    #     cada ventana de Q con su ventana más parecida en R
    #     recuerde guardar el nombre del archivo y los tiempos de inicio y fin que representa cada ventana de Q y R
    #     puede servir la funcion util.guardar_objeto() que está definida en util.py
    #
    util.escribir_lista_de_columnas_en_archivo(resultados_similares, archivo_ventanas_similares_txt)

    


# Main code to execute the search
if len(sys.argv) != 4:
    print(
        "Uso: {} [carpeta_descriptores_radio_Q] [carpeta_descritores_canciones_R] [archivo_ventanas_similares_txt]".format(
            sys.argv[0]))
    sys.exit(1)
# lee los parametros de entrada
carpeta_descriptores_radio_Q = sys.argv[1]
carpeta_descritores_canciones_R = sys.argv[2]
archivo_ventanas_similares_txt = sys.argv[3]

# llamar a la tarea
tarea2_busqueda(carpeta_descriptores_radio_Q, carpeta_descritores_canciones_R, archivo_ventanas_similares_txt)
