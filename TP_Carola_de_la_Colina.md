# # TP Manejo de Datos en Biología Computacional. Herramientas de Estadística

## descripción del sistema 
**•	Realizar una descripción del sistema que se intenta estudiar y de las variables medidas sobre la muestra:**
Las técnicas fluorométricas en células aisladas, utilizando sondas fluorescentes sensibles a Ca2+, permiten la medición cuantitativa de eventos dinámicos que ocurren en células vivas.
En este sistema se estudian celulas aisladas de corazon de rata previamente disgregadas. Se utiliza un fluoroforo sensible a Ca+2 para analizar la liberacion de Ca+2 del retículo sarcoplásmico, lo que da como resultado una senal que denominamos "spark". Se analiza la emision de senial a lo largo del tiempo por lo que tenemos informacion de parametros cineticos de cada senial. 
Por otro lado se tiene conocimiento de la presencia de senial que no se debe a liberacion de Ca+2. Estos eventos estan identificados en la columna "flag", donde 1 indica presencia de spark, y 2 indica ruido, o la presencia de senial que no es resultante de un spark.

Las variables incluidas en los datos utilizados son:
tiempo_maximo
intensidad_maxima
intensidad_minima
tiempo_valle
intensidad_valle
sparks_amplitud
TTP
sparks_tiempo_pico50
sp_tau
TTP50
fullWidth	
(ΔF/F0)/ΔTmax	
fullDuration	
width	
high	
flag

En este trabajo, de todas estas variables se utilizara:
tiempo_maximo: tiempo necesario para que la senial llegue a su maxima intensidad: Maxima senial detectada del fluoroforo
sp_tau: da informacion acerca de la velocidad necesaria para que el Ca+2 ingrese al RS
tiempo_valle: tiempo requerido para que la senial llegue al minimo de su valor
width:	
flag: variable categorica: 1 indica presencia de spark, y 2 indica senial por otra razon

## Ingreso de los datos 

```python

#importa los modulos a utilizar
import scipy.stats as ss
import statsmodels.stats.power as smp
import numpy as np
from statsmodels.stats.power import TTestIndPower
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#importa los datos
path = "C:/Users/Carola/Documents/Especializacion_bioinfo/ESTADISTICA/DataSets/tabla_trabajo_final.csv"
df_con_nan = pd.read_csv(path,sep=',')

```
## Tratamiento de NaN

```
df_con_nan.head()
df_con_nan.describe()
df_con_nan.shape

df_con_nan.isna().sum()

Unnamed: 0                0
tiempo_maximo            61
intensidad_maxima        61
intensidad_minima        61
tiempo_valle             61
intensidad_valle         61
sparks_amplitud          61
TTP                      61
sparks_tiempo_pico50     61
sp_tau                  328
TTP50                    61
fullWidth                61
(ΔF/F0)/ΔTmax            61
fullDuration             61
width                     0
high                      0
flag                      0
dtype: int64

En la tabla se muestra el numero de NaN que contiene cada una de las columnas. Se observa un grupo de 62 filas que poseen mas de 8 NaN. Considero que los datos obtenidos de estas filas provienen de mediciones erroneas y no aportan informacion a los datos. Es por esto que realice un filtro para luego eliminar estas filas del dataframe de trabajo.

#número de valores NaN en cada fila utilizando isna().sum(axis=1).
nan_por_fila = df_con_nan.isna().sum(axis=1) 

#filtrado de filas que tienen más de 5 valores NaN y se almacenan en la variable filas_a_eliminar
filas_a_eliminar = df_con_nan[nan_por_fila > 5] 

#Armo un df con las filas que van a ser eliminadas
eliminadas = df_con_nan[nan_por_fila > 5].copy() 

#elimina las filas con mas de 5 NaN
df = df_con_nan.drop(filas_a_eliminar.index)

#verifico la eliminacion de las filas
df.isna().sum()

Unnamed: 0                0
tiempo_maximo             0
intensidad_maxima         0
intensidad_minima         0
tiempo_valle              0
intensidad_valle          0
sparks_amplitud           0
TTP                       0
sparks_tiempo_pico50      0
sp_tau                  267
TTP50                     0
fullWidth                 0
(ΔF/F0)/ΔTmax             0
fullDuration              0
width                     0
high                      0
flag                      0
dtype: int64


```

## Representaciones de las variables a utilizar y de sus distribuciones 

```

df.hist(figsize=(12,12), bins = 30)
plt.show()

```

<img src=./imagenes/plot_distribuciones_columnas.png
     width="100%" 
     height=auto />

## Caracteristicas de las distribuciones 

## Estimacion los intervalos de confianza 


## Determinacion del tamanio de la muestra.    
**•	Realizar una descripción del sistema que se intenta estudiar y de las variables medidas sobre la muestra:**

## Ensayos de hipótesis:
**Realizar un contraste de hipótesis para dos o más poblaciones.**
**o	**Realizar un análisis de dependencia de variables categóricas.**
**o	**Evaluar el ajuste de una recta de regresión e interpretar el coeficiente de correlación.**

 







