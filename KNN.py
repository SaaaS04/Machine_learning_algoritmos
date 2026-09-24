import kagglehub
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt 
import os

path = kagglehub.dataset_download("bobbyscience/league-of-legends-diamond-ranked-games-10-min")

arquivos = os.listdir(path)
print("Arquivos no diretório:", arquivos)


arquivo_csv = os.path.join(path, "high_diamond_ranked_10min.csv")
partidas = pd.read_csv(arquivo_csv)

# colunas mais ligadas ao resultado
colunas = ["blueGoldDiff", "blueExperienceDiff", "blueKills",
           "blueDeaths", "blueDragons"]

atributos = partidas[colunas]
vitorias_blue_side = partidas["blueWins"]

print("Partidas:", len(partidas))
print(vitorias_blue_side.value_counts(normalize=True).round(3).to_string(), "\n")


# KNN usa distancia: precisa padronizar
# Pipeline refaz a padronizacao a cada fold
def modelo(k):
    return Pipeline([("escala", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=k))])


# estratificado mantem a proporcao das classes
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

vizinhos = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101]

acuracias = {}
print("  K | acuracia | desvio")
for k in vizinhos:
    folds = cross_val_score(modelo(k), atributos, vitorias_blue_side,
                            cv=cv, scoring="accuracy")
    acuracias[k] = folds.mean()
    print(f"{k:>3} |   {folds.mean():.4f} | {folds.std():.4f}")

melhor = max(acuracias, key=acuracias.get)
pior = min(acuracias, key=acuracias.get)

print(f"\nMelhor: K = {melhor} ({acuracias[melhor]:.4f})")
print(f"Pior:   K = {pior} ({acuracias[pior]:.4f})")
print(f"Diferenca: {(acuracias[melhor] - acuracias[pior]) * 100:.2f} pontos percentuais")
