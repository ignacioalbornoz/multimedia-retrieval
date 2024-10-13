# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [nombre]

import sys
import os
import util as util
import librosa




def tarea2_deteccion(archivo_ventanas_similares, archivo_detecciones):
    if not os.path.isfile(archivo_ventanas_similares):
        print("ERROR: no existe archivo {}".format(archivo_ventanas_similares))
        sys.exit(1)
    elif os.path.exists(archivo_detecciones):
        print("ERROR: ya existe archivo {}".format(archivo_detecciones))
        sys.exit(1)
    sample_rate = 22050
    n_mfcc = 20 
    n_fft = 2048
    hop_length = 512
    # Implementar la tarea con los siguientes pasos:
    #
    #  1-leer el archivo archivo_ventanas_similares (fue creado por tarea2_busqueda)
    #    puede servir la funcion util.leer_objeto() que está definida en util.py
    ventanas_similares = []
    with open(archivo_ventanas_similares, 'r') as f:
        for line in f:
            q_file, q_start_time, r_file, r_start_time, distancia, total_duracion_r = line.strip().split("\t")
            ventanas_similares.append([q_file, float(q_start_time), r_file, float(r_start_time), float(distancia), float(total_duracion_r)])

    #  2-crear un algoritmo para buscar secuencias similares entre audios
    #    ver slides de la semana 5 y 7
    #    identificar grupos de ventanas de Q y R que son similares y pertenecen a las mismas canciones con el mismo desfase
    detecciones = []
    confianza = 0
    for i in range(len(ventanas_similares)):
        

        q_file, q_start_time, r_file, r_start_time, distancia, total_duracion_r = ventanas_similares[i]
        # Duración total del archivo R 
        total_ventanas_r = total_duracion_r / (512 / 22050)  # Ventanas en R (basado en hop_length = 512 y sample_rate = 22050)
        
        # Initialize or reset detection sequence
        if i == 0 or q_file != ventanas_similares[i-1][0] or r_file != ventanas_similares[i-1][2]:
            # Start a new detection sequence
            q_start = q_start_time
            _ = r_start_time
            inicio_tiempo_q = q_start  # Start time of detection in Q
            desfase = q_start_time - r_start_time  # Constant time offset between Q and R
            confianza = 0  # Initialize confidence level
            
        # Check if time offset remains the same for consecutive windows
        if q_start_time - r_start_time == desfase:
            confianza += 1  # Increase confidence as more windows align
        else:
            confianza_normalizada = confianza / total_ventanas_r
            # Add completed detection sequence to the results
            q_file = q_file.replace('_mfcc.pkl', '.m4a')
            r_file = r_file.replace('_mfcc.pkl', '.m4a')
            detecciones.append([
                q_file,
                inicio_tiempo_q,  # Start time of detection in Q
                q_start_time - inicio_tiempo_q,  # Duration of detection in seconds
                r_file,
                confianza_normalizada  # Confidence level
            ])
            # Reset for the next sequence
            q_start = q_start_time
            _ = r_start_time
            inicio_tiempo_q = q_start
            desfase = q_start_time - r_start_time
            confianza = 1  # Reset confidence level

    # Add the last detection if applicable
    if confianza > 0:
        confianza_normalizada = confianza / total_ventanas_r
        q_file = q_file.replace('_mfcc.pkl', '.m4a')
        r_file = r_file.replace('_mfcc.pkl', '.m4a')
        detecciones.append([
            q_file,
            inicio_tiempo_q,
            q_start_time - inicio_tiempo_q,  # Duration of detection
            r_file,
            confianza_normalizada
        ])
    #  3-escribir las detecciones encontradas en archivo_detecciones, en un archivo con 5 columnas:
    #    columna 1: nombre de archivo Q (nombre de archivo en carpeta radio)
    #    columna 2: tiempo de inicio (número, tiempo medido en segundos de inicio de la emisión)
    #    columna 3: largo de la detección (número, tiempo medido en segundos con el largo de la emisión)
    #    columna 4: nombre de archivo R (nombre de archivo en carpeta canciones)
    #    columna 5: confianza (número, mientras más alto mayor confianza de la respuesta)
    #   le puede servir la funcion util.escribir_lista_de_columnas_en_archivo() que está definida util.py
    util.escribir_lista_de_columnas_en_archivo(detecciones, archivo_detecciones)
    print(f"Detección completada. Los resultados se han guardado en {archivo_detecciones}")



# inicio de la tarea
if len(sys.argv) != 3:
    print("Uso: {} [archivo_ventanas_similares] [archivo_detecciones]".format(sys.argv[0]))
    sys.exit(1)

# lee los parametros de entrada
archivo_ventanas_similares = sys.argv[1]
archivo_detecciones = sys.argv[2]

# llamar a la tarea
tarea2_deteccion(archivo_ventanas_similares, archivo_detecciones)
