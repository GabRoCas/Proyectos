import pickle
import pandas as pd


def cargar_datos_csv(path):
    return pd.read_csv(path)

def cargar_modelo(path):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model

def guardar_modelo(modelo):
    with open('finished_model.pkl', "wb") as file:
        pickle.dump(modelo, file)

def cargar_scaler(path):
    with open(path, "rb") as file:
        scaler = pickle.load(file)
    return scaler