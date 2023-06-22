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
## Ingreso de los datos 
```

df_con_nan.head()
df_con_nan.describe()
df_con_nan.shape

df_con_nan.isna().sum()


#número de valores NaN en cada fila utilizando isna().sum(axis=1).
nan_por_fila = df_con_nan.isna().sum(axis=1) 
#filtra filas que tienen más de 5 valores NaN y se almacenan en la variable filas_a_eliminar
filas_a_eliminar = df_con_nan[nan_por_fila > 5] 
#Armo un df con las filas que van a ser eliminadas
eliminadas = df_con_nan[nan_por_fila > 5].copy() 
 #elimina las filas con mas de 5 NaN
df = df_con_nan.drop(filas_a_eliminar.index)

df.isna().sum()


df.hist(figsize=(12,12), bins = 30)
plt.show()



```

## Representaciones de las variables a utilizar y de sus distribuciones 


## Caracteristicas de las distribuciones 

## Estimacion los intervalos de c¬onfianza 


## Determinacion del tamanio de la muestra.    
**•	Realizar una descripción del sistema que se intenta estudiar y de las variables medidas sobre la muestra:**

## Ensayos de hipótesis:
**Realizar un contraste de hipótesis para dos o más poblaciones.**
**o	**Realizar un análisis de dependencia de variables categóricas.**
**o	**Evaluar el ajuste de una recta de regresión e interpretar el coeficiente de correlación.**

 







