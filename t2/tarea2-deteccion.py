# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [nombre]

import sys
import os
import util as util
import librosa

class Candidato:
    def __init__(self, nombre, encontrados_totales, racha_no_encontrados, total_fotogramas, last_r_start_time, init_q_start_time, last_q_start_time, min_distance, init_r_start_time):
        self.nombre = nombre
        self.encontrados_totales = encontrados_totales
        self.racha_no_encontrados = racha_no_encontrados
        self.total_fotogramas = total_fotogramas
        self.last_r_start_time = last_r_start_time
        self.init_q_start_time = init_q_start_time
        self.last_q_start_time = last_q_start_time
        self.min_distance = min_distance
        self.init_r_start_time = init_r_start_time



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
            
            q_file, q_start_time, r_file, r_start_time, distancia, total_ventanas_r = line.strip().split("\t")
            ventanas_similares.append([q_file, float(q_start_time), r_file, float(r_start_time), float(distancia), float(total_ventanas_r)])

    unique_first_params = set(row[0] for row in ventanas_similares)

    for unique_param in sorted(unique_first_params):
        
        filtered_data = [row for row in ventanas_similares if row[0] == unique_param]
        sorted_data = sorted(filtered_data, key=lambda x: float(x[1]))

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
                # candidato[5] = init_q_start_time
                # candidato[6] = last_q_start_time
                # candidato[7] = min_distance
                # candidato[8] = init_r_start_time
                
                nuevo_candidato = Candidato(nombre=r_file,
                                            encontrados_totales=1,
                                            racha_no_encontrados=0,
                                            total_fotogramas=total_ventanas_r,
                                            init_r_start_time=r_start_time,
                                            last_r_start_time=r_start_time,
                                            init_q_start_time=q_start_time,
                                            last_q_start_time=q_start_time,
                                            min_distance=distancia)
                candidatos.append(nuevo_candidato)

        
            else:
                for candidato in candidatos:

                    if (r_file == candidato.nombre):
                    
                        if (r_start_time > candidato.last_r_start_time):
                            candidato.encontrados_totales+=1
                            candidato.racha_no_encontrados=0
                            candidato.last_r_start_time=r_start_time
                            candidato.last_q_start_time = q_start_time
                            
                            if distancia < candidato.min_distance:
                                candidato.min_distance = distancia
                            # promedio
                            # varianza


                        elif (r_start_time == candidato.last_r_start_time):
                            if(distancia < candidato.min_distance) and candidato.encontrados_totales==1:
                                candidatos.remove(candidato)
                                nuevo_candidato = Candidato(nombre=r_file,
                                                            encontrados_totales=1,
                                                            racha_no_encontrados=0,
                                                            total_fotogramas=total_ventanas_r,
                                                            init_r_start_time=r_start_time,
                                                            last_r_start_time=r_start_time,
                                                            init_q_start_time=q_start_time,
                                                            last_q_start_time=q_start_time,
                                                            min_distance=distancia)
                                candidatos.append(nuevo_candidato)
                            else:
                                candidato.racha_no_encontrados+=1

                            
                        elif (r_start_time < candidato.last_r_start_time):
                            if distancia < candidato.min_distance:
                                candidatos.remove(candidato)
                                nuevo_candidato = Candidato(nombre=r_file,
                                                            encontrados_totales=1,
                                                            racha_no_encontrados=0,
                                                            total_fotogramas=total_ventanas_r,
                                                            init_r_start_time=r_start_time,
                                                            last_r_start_time=r_start_time,
                                                            init_q_start_time=q_start_time,
                                                            last_q_start_time=q_start_time,
                                                            min_distance=distancia)
                                candidatos.append(nuevo_candidato)
                                
                            else:
                                candidato.racha_no_encontrados+=1

                    else:
                        candidato.racha_no_encontrados+=1
                    
                    if (candidato.racha_no_encontrados > candidato.total_fotogramas):
                        q_file_to_save = q_file.replace('_mfcc.pkl', '.m4a')
                        candidato.nombre = candidato.nombre.replace('_mfcc.pkl', '.m4a')
                       
                        confianza = (candidato.encontrados_totales/candidato.total_fotogramas) *  (1/candidato.min_distance)
                        start_time_detect_in_q = candidato.init_q_start_time - candidato.init_r_start_time
                        if ((candidato.last_q_start_time - candidato.init_q_start_time) > 0) and ((candidato.last_r_start_time - candidato.init_r_start_time) > 0):
                            detecciones.append([
                                q_file_to_save,
                                start_time_detect_in_q, 
                                candidato.total_fotogramas*0.1,  
                                candidato.nombre,
                                confianza  
                            ])
                        candidatos.remove(candidato)   
                
                if all(r_file != candidato.nombre for candidato in candidatos):
                    nuevo_candidato = Candidato(nombre=r_file,
                                                encontrados_totales=1,
                                                racha_no_encontrados=0,
                                                total_fotogramas=total_ventanas_r,
                                                init_r_start_time=r_start_time,
                                                last_r_start_time=r_start_time,
                                                init_q_start_time=q_start_time,
                                                last_q_start_time=q_start_time,
                                                min_distance=distancia)
                    candidatos.append(nuevo_candidato)

                #for ex_candidato in ex_candidatos:

        for candidato in candidatos:
            q_file_to_save = q_file.replace('_mfcc.pkl', '.m4a')
            candidato.nombre = candidato.nombre.replace('_mfcc.pkl', '.m4a')
            
            confianza = (candidato.encontrados_totales/candidato.total_fotogramas) *  (1/candidato.min_distance)
            start_time_detect_in_q = candidato.init_q_start_time - candidato.init_r_start_time

            if ((candidato.last_q_start_time - candidato.init_q_start_time) > 0) and ((candidato.last_r_start_time - candidato.init_r_start_time) > 0):
                detecciones.append([
                    q_file_to_save,
                    start_time_detect_in_q, 
                    candidato.total_fotogramas*0.1,  
                    candidato.nombre,
                    confianza  
                ])
            candidatos.remove(candidato)   



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
