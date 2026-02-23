# SVJ Predictor

Para este projecto se trabajo con la version Python 3.14.2. 
Pero puesto que nos encontramos con problemas, decidimos usar la version
3.11.3 de python para mitigar.
El proyecto realiza predicciones del precio en rango de 1H.

## Resolver enviroment phyton

Puede que tengamos problemas si ya tenemos la version mas reciente
de python por lo que cambiaremos a un entorno virtual con python 3.11.3
para eso debemos tenerlo ya descargado

### Pasos
 - Abriremos CMD e iremos a la ruta de nuestro programa
 - Crearemos el enterno con Python 3.11.3
 - 'py -3.11 -m venv .venv'
 - En Pycharm, se cambiara el interpreter en File -> Settings
 - Luego verificaremos en el terminal de pycharm 'python --version'

## Paquetes necesarios
  - numpy
  - scipy
  - pandas
  - yfinance
  - matplotlib
  - pymc
  - arviz
  - theano-pymc

## Calibracion Bayesian con PyMC

El metodo anteior y original fue desarrolado usando Nelder_Mead
en este branch usaremos Calibracion Bayesian, con la intension
de mejorar la exactitud de nuestro programa.

## Actualizacion, capacidad predictiva
### optimizador local -> Global

Se implementara Differential Evolution en vez de algoritmo 
L-BFGS-B.

### Refactorizacion de funcion perdida

Se noto que penalizaciones directas: loss =(...) + volume_penalty + rsi_penalty
confunde al optimizador al mezclar unidades distintas.

### Deteccion de Regimenes 
Filtro de Kalman o HMM, se agregara una logica de deteccion de cambio de regimen.

## Health Check

Rango SCI

85-100% Senal fuerte
60-84% Sennal normal
40-59% Riesgo Alto
<40 Basura, no confies en resultados.


