# *Análisis espectral de la voz*

## *Contexto historico*

Las características espectrales desempeñan un papel fundamental en el análisis y la comprensión de las señales de voz. Éstas capturan las características de frecuencia de los sonidos del habla, lo que permite comprender los patrones fonéticos, los rasgos del hablante y los matices lingüísticos. Algunas de las características espectrales clave que se utilizan habitualmente en el análisis de señales de voz son las siguientes:
- El centroide espectral: representa el centro de masa del espectro e indica la frecuencia promedio de la señal de voz. Ofrece información sobre el tono y el timbre percibidos de los sonidos del habla.

- La frecuencia fundamental (F0): es la frecuencia más baja y principal de un sonido, la que define su altura tonal.

- La frecuencia media: se refiere a una banda específica del espectro electromagnético o un rango específico de frecuencias.
  
- El Jitter y el shimmer: ambas son medidas de inestabilidad en señales periódicas (como la voz), pero se diferencian en lo que miden: jitter es la variación de la frecuencia (tono) ciclo a ciclo, mientras que shimmer es la variación de la amplitud (volumen) ciclo a ciclo, ambas causadas por la irregular vibración de las cuerdas vocales y percibidas como aspereza o temblor en la voz.

## *Objetivos*

### *Objetivo general:*
Emplear técnicas de análisis espectral para la diferenciación o clasificación de señales de voz según el género.

### *Objetivos especificos:*

- Capturar y procesar señales de voz masculinas y femeninas.

- Aplicar la Transformada de Fourier como herramienta de análisis espectral de la voz.

- Extraer parámetros característicos de la señal de voz: frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer.

- Comparar las diferencias principales entre señales de voz de hombres y mujeres a partir de su análisis en frecuencia.

- Desarrollar conclusiones sobre el comportamiento espectral de la voz humana en función del género.

## *Adquisición de datos*

| Parámetro                          | Valor                              |
|-----------------------------------|------------------------------------|
| Frecuencia de Muestreo (Fs)       | Hz                                 |
| Resolución                        |  bits                              |
| Nivel de entrada                  |                                    |
| Entorno                           |                                    |

***Nota:*** Se grabaron 6 señales (3 hombres, 3 mujeres) con la misma frase de ~3 segundos en formato .wav. 

## *Procesamiento de datos (codigo en Python)*

Este código tiene como objetivo realizar un análisis completo de señales de voz en formato `.wav`, permitiendo extraer características espectrales y calcular métricas importantes como el jitter y el shimmer, ampliamente utilizadas en el análisis biomédico de la calidad vocal.

Inicialmente, se importan las librerías necesarias para el procesamiento numérico, análisis de audio, visualización de señales, manejo de archivos y filtrado digital:

```python
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
from scipy.signal import butter, lfilter, find_peaks
```

A continuación, se define la lista de archivos de audio que serán procesados
```python
archivos = ['HOMBRE1.wav', 'HOMBRE2.wav', 'HOMBRE3.wav', 
            'MUJER1.wav', 'MUJER2.wav', 'MUJER3.wav']
```

Posteriormente, se implementa una función de filtro pasabanda tipo Butterworth, el cual permite conservar únicamente las frecuencias relevantes de la voz humana:
```python
def filtro_pasabanda(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)
```
En la Parte A, se carga la señal original y su frecuencia de muestreo:

```python
y_original, sr = librosa.load(nombre, sr=None)
```
Se calcula el número de muestras y el vector de tiempo:

```python
n = len(y_original)
t = np.linspace(0, n/sr, n)
```
Se clasifica el género del hablante:

```python
genero = 'Hombre' if 'HOM' in nombre.upper() else 'Mujer'
```
Se extraen las características principales de la señal.

- Frecuencia fundamental:

```python
f0 = np.nanmean(librosa.yin(y_original, fmin=80, fmax=450))
```
- Frecuencia media:
```python
f_media = np.mean(librosa.feature.spectral_centroid(y=y_original, sr=sr))
```
- Brillo:
```python
brillo = np.mean(librosa.feature.spectral_rolloff(y=y_original, sr=sr, roll_percent=0.85))
```
- Energía:
```python
energia = np.mean(librosa.feature.rms(y=y_original))
```
Se calcula la Transformada de Fourier para analizar la señal en el dominio de la frecuencia:
```python
fft_mag = np.abs(np.fft.rfft(y_original))
freqs = np.fft.rfftfreq(n, 1/sr)
```
En la Parte B, primero se filtra la señal según el rango de frecuencias de la voz:
```python
f_min, f_max = (80, 400) if genero == 'Hombre' else (150, 500)
y_filtrada_completa = filtro_pasabanda(y_original, f_min, f_max, sr)
```
Luego se selecciona un segmento específico de la señal filtrada:
```python
start, end = int(t_in * sr), int(t_out * sr)
y_filtrada_segmento = y_filtrada_completa[start:end]
```
Se normaliza la señal para evitar efectos de escala:
```python
y_seg_norm = y_filtrada_segmento / np.max(np.abs(y_filtrada_segmento))
```
Se detectan los picos correspondientes a cada ciclo de la señal:
```python
picos, _ = find_peaks(y_seg_norm, distance=sr/250, height=0.5)
```
A partir de los picos se calculan los periodos y amplitudes:
```python
Ti = np.diff(picos / sr)
Ai = y_seg_norm[picos]
```
Se calcula el jitter que mide la variación del periodo entre ciclos:
```python
jitter_abs = (1/(len(Ti)-1)) * np.sum(np.abs(Ti[:-1] - Ti[1:]))
jitter_rel = (jitter_abs / np.mean(Ti)) * 100
```
Y se calcula el shimmer que mide la variación de la amplitud entre ciclos:
```python
shimmer_abs = (1/(len(Ai)-1)) * np.sum(np.abs(Ai[:-1] - Ai[1:]))
shimmer_rel = (shimmer_abs / np.mean(Ai)) * 100
```
Finalmente, los resultados se almacenan en estructuras tipo DataFrame:
```python
df_a = pd.DataFrame(resultados_a)
df_b = pd.DataFrame(resultados_b)
```

## *Resultados*


A continuación se presentan las gráficas de las seis señales de voz en el dominio del tiempo.

<p align="center">
<img src="GRAFICAS/Graficas Hombre 1.png" width="600">
</p>
<p align="center">
<em>Gráfica 1. Gráficas para Hombre 1.</em>
</p>

| Parámetro              | Descripción |
|------------------------|-------------|
| Duración               |             |
| Presencia de silencios |             |
| Posibles artefactos    |             |

---

<p align="center">
<img src="GRAFICAS/Graficas Hombre 2.png" width="600">
</p>
<p align="center">
<em>Gráfica 2. Gráficas para Hombre 2.</em>
</p>

| Parámetro              | Descripción |
|------------------------|-------------|
| Duración               |             |
| Presencia de silencios |             |
| Posibles artefactos    |             |

---

<p align="center">
<img src="GRAFICAS/Graficas Hombre 3.png" width="600">
</p>
<p align="center">
<em>Gráfica 3. Gráficas para Hombre 3.</em>
</p>

| Parámetro              | Descripción |
|------------------------|-------------|
| Duración               |             |
| Presencia de silencios |             |
| Posibles artefactos    |             |

---

<p align="center">
<img src="GRAFICAS/Graficas Mujer 1.png" width="600">
</p>
<p align="center">
<em>Gráfica 4. Gráficas para Mujer 1.</em>
</p>

| Parámetro              | Descripción |
|------------------------|-------------|
| Duración               |             |
| Presencia de silencios |             |
| Posibles artefactos    |             |

---

<p align="center">
<img src="GRAFICAS/Graficas Mujer 2.png" width="600">
</p>
<p align="center">
<em>Gráfica 5. Gráficas para Mujer 2.</em>
</p>

| Parámetro              | Descripción |
|------------------------|-------------|
| Duración               |             |
| Presencia de silencios |             |
| Posibles artefactos    |             |

---

<p align="center">
<img src="GRAFICAS/Graficas Mujer 3.png" width="600">
</p>
<p align="center">
<em>Gráfica 6. Gráficas para Mujer 3.</em>
</p>

| Parámetro              | Descripción |
|------------------------|-------------|
| Duración               |             |
| Presencia de silencios |             |
| Posibles artefactos    |             |


- ***Extracción de características por señal:***

A continuación se presentan los valores calculados para cada señal de voz.

| Archivo   | Frecuencia fundamental (F0) | Frecuencia media o Brillo (centroide espectral) | Intensidad / energía (RMS) |
|-----------|----------------------------|--------------------------------------------------|-----------------------------|
| Hombre 1  |                            |                                                  |                             |
| Hombre 2  |                            |                                                  |                             |
| Hombre 3  |                            |                                                  |                             |
| Mujer 1   |                            |                                                  |                             |
| Mujer 2   |                            |                                                  |                             |
| Mujer 3   |                            |                                                  |                             |

- ***Medición de estabilidad vocal:***

- ***Comparación hombres vs. mujeres:***

-***Interpretación clínica/técnica:***

## *Comparación y conclusiones*
Comparar los resultados obtenidos entre las voces masculinas y femeninas.

1. ¿Qué diferencias se observan en la frecuencia fundamental?
   
2. ¿Qué otras diferencias notan en términos de brillo, media o intensidad?
   
3. Redactar conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados.

5. Discuta la importancia clínica del jitter y shimmer en el análisis de la voz.
