 # TP Manejo de Datos en Biología Computacional y Herramientas de Estadística

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
### Tratamiento de NaN

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
     width="80%" 
     height=auto />

## Creo los dataframes df_1 y df_2
**•	Armo un dataframe para cada conjunto df_1 y df_2, dpendiendo de su valor en la columna "flag"**

```python

df_1 = df[df["flag"] == 1]
df_2 = df[df["flag"] == 2]

```

## Tratamiento de outlayers de df_1 y df_2
Inicialmente realizo de bloques para visualizar los datos de df_1 y df_2

<img src=./imagenes/outlayers_1.png
     width="40%" 
     height=auto />

<img src=./imagenes/outlayers_2.png
     width="40%" 
     height=auto />

Para el tratamiento de outlayers se descartaron los datos que estan en los percentilos 2 y 98 de las tablas df_1 y df_2:

```python

Q1 = df_1["width"].quantile(0.05)
Q3 = df_1["width"].quantile(0.95)
IQR = Q3 - Q1
df_1_sin_outliers = df_1[(df_1["width"] >= Q1 - 1.5 * IQR) & (df_1["width"] <= Q3 + 1.5 * IQR)]

Q1 = df_2["width"].quantile(0.05)
Q3 = df_2["width"].quantile(0.95)
IQR = Q3 - Q1
df_2_sin_outliers = df_2[(df_2["width"] >= Q1 - 1.5 * IQR) & (df_2["width"] <= Q3 + 1.5 * IQR)]

```

En este código, Q1 y Q3 representan el percentil 2 y 98 de la columna "width", respectivamente. Calculo el rango intercuartílico (IQR) como la diferencia entre Q3 y Q1.
Se defino idx: variable que identifica los valores que están por debajo de (Q1 - 1.5 * IQR) o por encima de (Q3 + 1.5 * IQR). 
Finalmente, se utilizo idx para filtrar los DataFrames  df_1 y df_2 y se les asigna el resultado a df_1_sin_outliers y df_2_sin_outliers.

Utilizando la funcion shape se analizo la cantidad de datos que fueron descartados. En este caso no hubo valores fuera de los limites propuestos por lo que no se eliminaron datos.

Utilizando la funcion .head() se verifico que las tablas df_1_sin_outliers y df_2_sin_outliers se hubieran definido correctamente.

## Estimacion los intervalos de confianza 

El cálculo del intervalo de confianza utiliza la distribución t de Student para estimar un rango dentro del cual se espera que se encuentre la media poblacional con un nivel de confianza determinado.

```python

confidence_level = 0.95
confidence_intervals = {}

mean1 = np.mean(df_1_sin_outliers["width"]) #intervalo para flag1
n1 = len(df_1_sin_outliers["width"])
std_error1 = np.std(df_1_sin_outliers["width"], ddof=1) / np.sqrt(n1)
margin_of_error1 = std_error1 * stats.t.ppf((1 + confidence_level) / 2, n1 - 1)
confidence_interval_1 = (mean1 - margin_of_error1, mean1 + margin_of_error1)
    
mean2 = np.mean(df_2_sin_outliers["width"]) #intervalo para flag2
n2 = len(df_2_sin_outliers["width"])
std_error2 = np.std(df_2_sin_outliers["width"], ddof=1) / np.sqrt(n2)
margin_of_error2 = std_error2* stats.t.ppf((1 + confidence_level) / 2, n1 - 1)
confidence_interval_2 = (mean1 - margin_of_error2, mean1 + margin_of_error2)
    
print(confidence_interval_1)  

```
(15.268263363694762, 15.858342140892395)

```python
print(confidence_interval_2)   

```
(15.174704855186013, 15.951900649401145)


## Caracteristicas de las distribuciones 

Se analizaron las caracteristicas generales de las distribuciones de la columna "width" en ambas tablas:

```python

df_1_sin_outliers["width"].describe()

```

<table>
<tr>
<td>count</td>
<td>1090.000000</td>
</tr>
<tr>
<td>mean</td>
<td>15.563303</td>
</tr>
<tr>
<td>std</td>
<td>4.964348</td>
</tr>
<tr>
<td>min</td>
<td>7.000000</td>
</tr>
<tr>
<td>25%</td>
<td>12.000000</td>
</tr>
<tr>
<td>50%</td>
<td>15.000000</td>
</tr>
<tr>
<td>75%</td>
<td>18.000000</td>
</tr>
<tr>
<td>max</td>
<td>42.000000</td>
</tr>
<tr>
<td>max</td>
<td>42.000000</td>
</tr>
</table>


```python

df_1_sin_outliers["width"].describe()

```
<table>
<tr>
<td>count</td>
<td>1246.000000</td>
</tr>
<tr>
<td>mean</td>
<td>14.262440</td>
</tr>
<tr>
<td>std</td>
<td>6.990825</td>
</tr>
<tr>
<td>min</td>
<td>4.000000</td>
</tr>
<tr>
<td>25%</td>
<td>9.000000</td>
</tr>
<tr>
<td>50%</td>
<td>12.000000</td>
</tr>
<tr>
<td>75%</td>
<td>17.000000</td>
</tr>
<tr>
<td>max</td>
<td>53.000000</td>
</tr>
<tr>
<td>max</td>
<td>42.000000</td>
</tr>
</table>

## Ensayos de hipótesis para la columna "width" en ambas tablas:

Inicialmente se aplico el shapiro test para evaluar si los datos siguen una distribucion normal.
Para esto primero se calcula el tamanio muestral necesario

```python

from statsmodels.stats.power import tt_ind_solve_power
effect_size = 0.5  # Tamaño del efecto esperado
alpha = 0.05  # Nivel de significancia (probabilidad de cometer un error tipo I)
power = 0.8  # Potencia (1 - probabilidad de cometer un error tipo II)

sample_size = tt_ind_solve_power(effect_size=0.5, alpha=0.05, power=0.8)

print("Tamaño muestral necesario:", int(sample_size))

```
Tamaño muestral necesario: 63

Con el resultado del calculo del tamanio muestral verificamos que es correcto aplicar el test shapiro:

```python

def aplicar_shapiro_test(column):
    statistic, pvalue = ss.shapiro(column)
    if pvalue > 0.05:
        print("la dist de es normal: pvalue =", pvalue)
    else: 
        print("la dist no es normal: pvalue =", pvalue)

aplicar_shapiro_test(df_1_sin_outliers["width"])
aplicar_shapiro_test(df_2_sin_outliers["width"])

```

Los pvalue obtenidos para la columna "width" son:

df_1_sin_outliers: 1.1941276258043541e-20
df_2_sin_outliers: 5.590924067605381e-33

Dado que ambos valores son menores que el punto de corte 0.05, se puede afirmar que las distribuciones de ambos grupos no siguen una distribucion normal. 

A partir de este resultado se decidio continuar el analisis con tecnicas no parametricas, las cuales son mas apropiadas para conjuntos que no tienen un comportamiento normal.

### MAN WITNEY

Es una prueba de rangos con signos, es un test no paramétrico utilizado para determinar si hay una diferencia significativa entre dos grupos independientes en una variable continua.

Se utiliza cuando no se cumplen los supuestos de normalidad o igualdad de varianzas necesarios para realizar una prueba t de Student. En su lugar, se basa en los rangos de los datos para evaluar si hay una diferencia significativa entre los dos grupos.

Su objetivo es determinar si las muestras de dos grupos provienen de la misma población o si tienen una distribución de valores significativamente diferente. La hipótesis nula (H0) establece que no hay diferencia entre los grupos, mientras que la hipótesis alternativa (H1) establece que hay una diferencia significativa.

Requicitos para aplicar el test
#Asumir que las distribuciones tienen la misma forma 
#Los datos tienen que ser independientes.
#Los datos tienen que ser ordinales o bien se tienen que poder ordenarse de menor a mayor.
#No es necesario asumir que las muestras se distribuyen de forma normal o que proceden de poblaciones normales. Sin embargo, para que el test compare medianas, ambas han de tener el mismo tipo de distribución (varianza, asimetría, ...).
#Igualdad de varianza entre grupos.
```python

def aplicar_mannwhitneyu(col1, col2):
    statistic, pvalue = ss.mannwhitneyu(col1, col2, use_continuity=True, alternative='two-sided', axis=0, method='auto', nan_policy='propagate', keepdims=False)
    if pvalue > 0.05:
        print("No hay una diferencias significativas entre las medias de las distribuciones de los grupos : pvalue =", pvalue)
    else: 
        print("Hay diferencias significativas entre las medias de las distribuciones de los grupos", pvalue)

aplicar_mannwhitneyu(df_1_sin_outliers["width"],  df_2_sin_outliers["width"])

```
Hay diferencias significativas entre las medias de las distribuciones de los grupos 1.0946729734710784e-22



Analisis de la igualdad de varianzas: test de Levene
```python

from scipy.stats import ks_2samp
import scipy.stats as stats

def aplicar_levene(col1, col2):
    statistics, pvalue = stats.levene(col1, col2)
    if pvalue > 0.05:
       print("No existe una diferencia significativa entre las varianzas de los grupos, pvalue =", pvalue)
    else:
       print("Existe una diferencia significativa entre las varianzas de los grupos =", pvalue)
     
aplicar_levene(df_1_sin_outliers["width"],  df_2_sin_outliers["width"])

```
Existe una diferencia significativa entre las varianzas de los grupos = 2.5143639144404683e-09
No fue correcto aplicar el test de Man Witney



###	Kolmogorov

Permite verificar si las puntuaciones de la muestra siguen o no una distribución normal.
Es una prueba de bondad de ajuste: sirve para contrastar la hipótesis nula de que la distribución de una variable se ajusta a una determinada distribución teórica de probabilidad. 


```python

def aplicar_Kolmogorov(col1, col2):
    estadistico, pvalue = ks_2samp (col1, col2)
    if pvalue > 0.05:
       print("los conjuntos provienen de la misma distribucion =", pvalue)
    else:
       print("los dos conjuntos de datos de muestra no provienen de la misma distribución =", pvalue)       

aplicar_Kolmogorov(df_1_sin_outliers["width"],  df_2_sin_outliers["width"]) 

```
los dos conjuntos de datos de muestra no provienen de la misma distribución = 3.993346943681931e-27

un valor p tan pequeño sugiere fuertemente que hay una diferencia significativa entre 
los grupos que estás comparando. Por lo tanto, puedes concluir que existe evidencia 
estadística sólida para rechazar la hipótesis nula y afirmar que los grupos difieren 
de manera significativa en la variable que se está analizando

###	wilcoxon

La prueba de los rangos con signo de Wilcoxon es una prueba no paramétrica para comparar el rango medio de dos muestras relacionadas y determinar si existen diferencias entre ellas. Se utiliza como alternativa a la prueba t de Student cuando no se puede suponer la normalidad de dichas muestras.
Para realizar el test de wilcoxon se debe tener columnas con igual largo. 
Es por esto que se realizo una reduccion aleatoria del numero de filas para que el largo este igualado
Hipótesis nula: Las medianas de las dos columnas df_1_sin_outliers["width"] y df_2_sin_outliers_red["width"] son iguales.

```python

f1,c = df_1_sin_outliers.shape
f2,c = df_2_sin_outliers.shape
print("filas de df_1:",f1,"filas de df_2:",f2)
if f1>f2:
    df_1_sin_outliers_red= df_1_sin_outliers.sample(f2)
    f3,c = df_1_sin_outliers_red.shape
    print("se redujo df_1 a", f3)
    print("df_2:", f2)
if f1<f2:
    df_2_sin_outliers_red= df_2_sin_outliers.sample(f1)
    f4,c = df_2_sin_outliers_red.shape
    print("se redujo df_2 a", f4)
    print("df_1: ", f1)
else:
    pass

```
filas de df_1: 1090 filas de df_2: 1246
se redujo df_2 a 1090
df_1:  1090


```python

def aplicar_wilcoxon(col1, col2):
    
    statistic, pvalue = ss.wilcoxon(col1, col2,zero_method='wilcox', correction=False, alternative='two-sided', method='auto', axis=0, nan_policy='propagate', keepdims = False)
    if pvalue > 0.05:
        print("No hay una diferencias significativas entre la media de los grupos : pvalue =", pvalue)
    else: 
        print("Hay diferencias significativas entre la media de los grupos, pvalue=", pvalue)

aplicar_wilcoxon(df_1_sin_outliers["width"], df_2_sin_outliers_red["width"])

```
Hay diferencias significativas entre la media de los grupos, pvalue= 1.3795423540485293e-11

Esto sugiere que hay evidencia suficiente para rechazar la hipótesis nula y concluir que hay diferencias significativas entre las medianas de las dos muestras.

Calculo del tamanio muestral necesario para realizar la prueba de wilcoxon:
```python

from scipy.stats import wilcoxon

# Parámetros para el cálculo del tamaño muestral
effect_size = 0.5  # Tamaño del efecto esperado
alpha = 0.05  # Nivel de significancia
power = 0.8  # Poder estadístico deseado

# Función para calcular el tamaño muestral
def calculate_sample_size(effect_size, alpha, power):
    n = 2  # Inicialización del tamaño muestral
    while True:
        _, p_value = wilcoxon([0] * n, [effect_size] * n)
        if p_value < alpha or n >= 100000:
            break
        n += 1
    return n

# Cálculo del tamaño muestral
sample_size = calculate_sample_size(effect_size, alpha, power)

# Imprimir el tamaño muestral calculado
print("Tamaño muestral:", sample_size)

```
Tamaño muestral: 6
fue correcto aplicar el test


Como conclusion luego de aplicar los test no parametricos de Kolmogorov y Wilcoxon se puede inferir que los datos de las columnas "width" en las dos poblaciones analizadas corresponden a diferentes distribuciones, por lo que se encuentra una diferencia significativa entre las medias.

**Realizar un análisis de dependencia de variables categóricas.**

Para tener una idea general de las correlaciones entre las diferentes variables realice una tabla de correlaciones entre todas las columnas

```python

corr = df[df.columns].corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True)
sns.set(rc={'figure.figsize':(34,24)})

```

<img src=./imagenes/correlacion.png
     width="100%" 
     height=auto />


## Tabla de contingencia: relacion entre "flag" y presencia de NaN en la columna "sp_tau"

Se utilizo la columna sp_tau porque es la que presenta valores NaN luego del filtrado inicial.
Para realizar esta tabla se armaron dos listas con el fin de categorizar los datos: 
is_spark: contiene informacion acerca de si la senial se debe a un spark o a otro tipo de senial
is_nan: indica si los datos de la columna "sp_tau" contienen NaN o no.

```python
import numpy as np
import math

is_spark = []
for item in df["flag"]:
    if item == 1:
        is_spark.append(True)
    elif item == 2:
        is_spark.append(False)    


is_nan = []
for item in df["sp_tau"]:
    if math.isnan(item):
        is_nan.append(False)
    else:
        is_nan.append(True)

```

Luego se creo un dataframe con las listas armadas previamente:

```python

data = {
    'is_spark': is_spark,
    'is_nan': is_nan
}        

df_contingencia = pd.DataFrame(data)

```
Por ultimo se armo la tabla de contingencia y se analizo mediante la prueba de chi-cuadrado (χ²) con el objetivo de evaluar la asociación entre las dos variables categóricas.

```python

a = df_contingencia ['is_spark'] == True
b = df_contingencia ['is_nan'] == True
groups = df_contingencia.groupby([a,b]).count() 

#Convertir la tabla de contingencia a un array
observed = groups.values

chi2, pvalue = ss.chisquare(groups, ddof=0, axis=0)

print("chi2 statistics", chi2)
print("pvalue", pvalue)

```


## Tabla de contingencia: relacion entre "tiempo_valle" y "width".

Se categorizando las columnas "width" y "tiempo_valle", dependiendo de si su valor se encuentra por encima o por debajo de las medias calculadas

```python
import numpy as np
import math

whidth_under_mean = []
for value in df["width"]:
    if value <= 14.815603:
        whidth_under_mean.append(True)
    elif value > 14.815603:
        whidth_under_mean.append(False)    

tiempo_valle_under_mean = []
for value in df["tiempo_valle"]:
    if value <= 10.145120:
        tiempo_valle_under_mean.append(True)
    elif value > 10.145120:
        tiempo_valle_under_mean.append(False)  


```

Luego se creo un dataframe df_contingencia_ con las listas armadas previamente:

```python

#arme un df con las listas creadas anteriormente       
data_ = {
    'whidth_under_mean':whidth_under_mean,
    'tiempo_valle_under_mean': tiempo_valle_under_mean
}        

df_contingencia_ = pd.DataFrame(data_)

```

Luego se utilizo group by para comparar los datos en una tabla de contingencia.
Por ultimo se realizo un analisis mediante la prueba de chi-cuadrado (χ²) con el objetivo de evaluar la relacion entre las dos variables categóricas.

```python

tabla_contingencia_ = pd.crosstab(df_contingencia_['whidth_under_mean'], df_contingencia_['tiempo_valle_under_mean'])
print("tabla_contingencia: ")
print(tabla_contingencia_)
a = df_contingencia_['whidth_under_mean']
b = df_contingencia_['tiempo_valle_under_mean']

#comparamos los datos en una tabla de contingencia
groups = df_contingencia_.groupby([a,b]).count() 
print("print groups:")
print (groups)

print(ss.chisquare(groups, ddof=0, axis=0))


```

**Evaluar el ajuste de una recta de regresión e interpretar el coeficiente de correlación.**

Para determinar si existe una asociación entre dos variables y la fuerza de esta asociación se calculo el coeficiente de correlación de Pearson: Es un test paramétrico que se utiliza para medir 
la relación lineal entre dos variables continuas. El coeficiente de correlación de Pearson
toma valores entre -1 y 1, donde -1 indica una correlación negativa perfecta, 0 indica ausencia de correlación, y 1 indica una correlación positiva perfecta.

```python

from scipy.stats import pearsonr

corr, p_value = pearsonr(df_1_sin_outliers["width"], df_1_sin_outliers["tiempo_maximo"])

print("Coeficiente de correlación:", corr)
print("Valor p:", p_value)

```
Coeficiente de correlación: 0.7834139992806171
Valor p: 5.667521695345137e-227

El coeficiente de correlación de 0.7834 da indicios de que existe una correlación entre las variables "width" y "tiempo_maximo". Para evaluar si este resultado es significativo se analiza el valor p. En este caso el valor p es mucho menor que 0.05 lo que indica que la relación observada no es azarosa.

Por ultimo se realizo un grafico de la relacion entre las dos variables analizadas previamente

```python

X = df_1_sin_outliers["width"]
Y = df_1_sin_outliers["tiempo_maximo"]

plt.scatter(X, Y)
plt.xlabel("width")
plt.ylabel("tiempo_maximo")
plt.title("Gráfico de dispersión: width vs tiempo_maximo")
plt.show()

```
<img src=./imagenes/grafico_dispersion.png
     width="50%" 
     height=auto />

