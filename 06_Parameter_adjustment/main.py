import pandas as pd

# Llibreries que necessitarem
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

random_value = 33
# Carrega de dades i preparació de les dades emprant Pandas
data = pd.read_csv("data/day.csv")
datos = pd.DataFrame(data.iloc[:, 4:13])  # Seleccionam totes les files i les columnes per index
valors = pd.DataFrame(data.iloc[:, -1])  # Seleccionam totes les files i la columna objectiu

X = datos.to_numpy()
y = valors.to_numpy().ravel()
features_names = datos.columns
