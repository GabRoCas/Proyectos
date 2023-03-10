## Cargar librerías
import sys, os
sys.path.append(os.path.realpath('./utils'))
import pandas as pd
import pickle
import numpy as np 

from funciones import *
from xgboost import XGBClassifier

## Cargar datos
from datos import *
new_data = datos 
data = datos_modelo

## Cargar Modelo
modelo = cargar_modelo('/Users/gabrielarodrigocastrillo/Desktop/The_Bridge/Data_Science/COPIA_Repositorio/03-Machine_Learning/Proyecto_ML/SRC/finished_model.pkl')

## Entrenar Modelo - en caso de no tener que entrenar (o reentrenar) saltar este paso
#retrined_model = modelo.fit(new_data) - yo lo he entrenado en el notebook

## Predecir   
predictions = modelo.predict(data)

## Guardar modelo
guardar_modelo(modelo)

## Guardar predicciones - idealmente en BBDD
predictions.to_csv('new_preds.csv') 
