import numpy as np
import sys, os
sys.path.append(os.path.realpath('./utils'))
import pandas as pd
import pickle
from funciones import *

path = '/Users/gabrielarodrigocastrillo/Desktop/The_Bridge/Data_Science/COPIA_Repositorio/03-Machine_Learning/Proyecto_ML/SRC/data/final/final_data.csv'
datos = cargar_datos_csv(path) #cuando sólo tenga test 
datos_modelo = datos.drop(columns='target') #en mi caso porque los datos que tengo son todos de train

## Limpieza datos
def one_hot(df, column, prefix):
    oh = pd.get_dummies(df, columns = column, prefix = prefix)
    return oh

def transform_log(df):
    log = np.log(df.column)
    return log

## SCALER
scaler = cargar_scaler('/Users/gabrielarodrigocastrillo/Desktop/The_Bridge/Data_Science/COPIA_Repositorio/03-Machine_Learning/Proyecto_ML/SRC/scaler.pkl')

## PCA
pca = cargar_modelo('/Users/gabrielarodrigocastrillo/Desktop/The_Bridge/Data_Science/COPIA_Repositorio/03-Machine_Learning/Proyecto_ML/SRC/pca_model.pkl')

## K-MEANS
kmeans = cargar_modelo('/Users/gabrielarodrigocastrillo/Desktop/The_Bridge/Data_Science/COPIA_Repositorio/03-Machine_Learning/Proyecto_ML/SRC/kmeans_model.pkl')
