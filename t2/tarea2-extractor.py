# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [nombre]

import sys
import os
import util as util
import librosa
import numpy as np

def tarea2_extractor(carpeta_audios_entrada, carpeta_descriptores_salida):
    if not os.path.isdir(carpeta_audios_entrada):
        print("ERROR: no existe {}".format(carpeta_audios_entrada))
        sys.exit(1)
    elif os.path.exists(carpeta_descriptores_salida):
        print("ERROR: ya existe {}".format(carpeta_descriptores_salida))
        sys.exit(1)

    # Implementar la tarea con los siguientes pasos:
    #
    #  1-leer los archivos con extension .m4a que están carpeta_audios_entrada
    #    puede servir la funcion util.listar_archivos_con_extension() que está definida en util.py
    archivos_m4a = util.listar_archivos_con_extension(carpeta_audios_entrada, ".m4a")
    
    #  2-convertir cada archivo de audio a wav (guardar los wav temporales en carpeta_descriptores_salida)
    #    puede servir la funcion util.convertir_a_wav() que está definida en util.py

    #sample_rate = 22050
    sample_rate = 22050
    n_mfcc = 30
    n_fft = 2205
    hop_length = 2205
    for archivo_m4a in archivos_m4a:
        archivo_completo = os.path.join(carpeta_audios_entrada, archivo_m4a)
        
        archivo_wav = util.convertir_a_wav(archivo_completo, sample_rate, carpeta_descriptores_salida)
        
        
        samples, sr = librosa.load(archivo_wav, sr=sample_rate)
        samples = librosa.util.normalize(samples)  # Normalize the audio to handle volume variations
        

        mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Apply z-score normalization to MFCCs
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
        
        # Compute Chroma feature
        chroma = librosa.feature.chroma_stft(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma = (chroma - np.mean(chroma, axis=1, keepdims=True)) / np.std(chroma, axis=1, keepdims=True)

        #mfccs = mfccs.T
        # Concatenate spectral features with MFCC and Chroma
        combined_features = np.concatenate((mfccs, chroma), axis=0)
        combined_features = combined_features.T
        
        #  4-escribir en carpeta_descriptores_salida los descriptores de cada archivo
        #    puede servir la funcion util.guardar_objeto() que está definida en util.py
    
        nombre_salida = archivo_m4a.replace('.m4a', '_mfcc.pkl')
        util.guardar_objeto(combined_features, carpeta_descriptores_salida, nombre_salida)



# inicio de la tarea
if len(sys.argv) != 3:
    print("Uso: {} [carpeta_audios_entrada] [carpeta_descriptores_salida]".format(sys.argv[0]))
    sys.exit(1)

# lee los parametros de entrada
carpeta_audios_entrada = sys.argv[1]
carpeta_descriptores_salida = sys.argv[2]

# llamar a la tarea
tarea2_extractor(carpeta_audios_entrada, carpeta_descriptores_salida)
