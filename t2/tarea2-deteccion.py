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


    detecciones = [] 

    ventanas_similares = []
    with open(archivo_ventanas_similares, 'r') as f: 
        for line in f:
            # Dividimos cada línea por tabulaciones
            q_file, q_start_time, r_file, r_start_time, distancia, total_ventanas_r = line.strip().split("\t")
            
            # Añadimos los valores procesados a la lista, convirtiendo los campos necesarios a float
            ventanas_similares.append([q_file, float(q_start_time), r_file, float(r_start_time), float(distancia), float(total_ventanas_r)])

    # Obtén los valores únicos del primer parámetro
    unique_first_params = set(row[0] for row in ventanas_similares)

    # Itera sobre cada valor único del primer parámetro
    for unique_param in sorted(unique_first_params):
        # Filtra las filas donde el primer parámetro es igual al valor único actual
        filtered_data = [row for row in ventanas_similares if row[0] == unique_param]
        
        # Ordena las filas por el segundo parámetro (el índice 1 de cada fila)
        sorted_data = sorted(filtered_data, key=lambda x: float(x[1]))
        
        # Imprime el encabezado para cada grupo de datos
        print(f"\nDatos para: {unique_param}")

        #  2-crear un algoritmo para buscar secuencias similares entre audios
        #    ver slides de la semana 5 y 7
        #    identificar grupos de ventanas de Q y R que son similares y pertenecen a las mismas canciones con el mismo desfase
        candidatos = []
        #ex_candidatos = [] 
        #confianza = 0
        for row in sorted_data:
            q_file, q_start_time, r_file, r_start_time, distancia, total_ventanas_r = row

            if candidatos == []:
                # candidato[0] = nombre
                # candidato[1] = encontrados_totales
                # candidato[2] = racha_no_encontrados
                # candidato[3] = total_fotogramas   
                # candidato[4] = last_r_start_time
                # candidato[5] = last_q_start_time
                # candidato[6] = last_q_end_time
                # candidato[7] = min_distancia
                #
                #
                #
                #

                nuevo_candidato = [r_file, 1, 0, total_ventanas_r, r_start_time, q_start_time, q_start_time, distancia]
                candidatos.append(nuevo_candidato)
        
            else:
                for candidato in candidatos:
                    nombre=candidato[0]
                    #encontrados_totales=candidato[1]
                    #racha_no_encontrados=candidato[2]
                    #total_fotogramas=candidato[3]
                    last_r_start_time=candidato[4]

                    if (r_file == nombre):
                    
                        if (r_start_time > last_r_start_time):
                            candidato[1]+=1
                            candidato[2]=0
                            candidato[4]=r_start_time
                            candidato[6] = q_start_time

                        elif (r_start_time == last_r_start_time):
                            candidato[2]+=1
                            '''
                            candidato[1]+=1                     
                            candidato[2]=0
                            candidato[4]=r_start_time
                            candidato[6] = q_start_time
                            '''
                        elif (r_start_time < last_r_start_time):
                            if distancia < candidato[7]:
                                candidatos.remove(candidato)
                                nuevo_candidato = [r_file, 1, 0, total_ventanas_r, r_start_time, q_start_time, q_start_time, distancia]
                                candidatos.append(nuevo_candidato)
                                
                            else:
                                candidato[2]+=1

                    else:
                        candidato[2]+=1
                    
                    if (candidato[2] > candidato[3]):
                        #ex_candidatos.append(candidato)
                        #candidatos.remove(candidato)
                        q_file_to_save = q_file.replace('_mfcc.pkl', '.m4a')
                        candidato[0] = candidato[0].replace('_mfcc.pkl', '.m4a')
                        confianza = (candidato[1]/candidato[3] ) * (1/candidato[7])
                        if candidato[6] - candidato[5] > 0:
                            detecciones.append([
                                q_file_to_save,
                                candidato[5],  # Start time of detection in Q
                                candidato[6] - candidato[5],  # Duration of detection in seconds
                                candidato[0],
                                confianza  # Confidence level
                            ])
                        candidatos.remove(candidato)   
                
                if all(r_file != candidato[0] for candidato in candidatos):

                    nuevo_candidato = [r_file, 1, 0, total_ventanas_r, last_r_start_time, q_start_time, q_start_time, distancia]
                    candidatos.append(nuevo_candidato)

                #for ex_candidato in ex_candidatos:

        for candidato in candidatos:
            q_file_to_save = q_file.replace('_mfcc.pkl', '.m4a')
            candidato[0] = candidato[0].replace('_mfcc.pkl', '.m4a')
            confianza = (candidato[1]/candidato[3] ) * (1/candidato[7])
            if (candidato[6] - candidato[5] > 0):
                detecciones.append([
                    q_file_to_save,
                    candidato[5],  # Start time of detection in Q
                    candidato[6] - candidato[5],  # Duration of detection in seconds
                    candidato[0],
                    confianza  # Confidence level
                ])
            candidatos.remove(candidato)   

            



    # HASTA ACA


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
