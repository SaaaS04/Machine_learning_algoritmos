import kagglehub
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt 
import os

caminho_dataset = kagglehub.dataset_download("bobbyscience/league-of-legends-diamond-ranked-games-10-min")

arquivos_dataset = os.listdir(caminho_dataset)
print("Arquivos no diretório:", arquivos_dataset)


arquivo_csv = os.path.join(caminho_dataset, "high_diamond_ranked_10min.csv")
partidas = pd.read_csv(arquivo_csv)

# tratamento: remove nulos e partidas duplicadas
print("Nulos:", partidas.isnull().sum().sum())
print("Duplicadas:", partidas.duplicated(subset="gameId").sum())
partidas = partidas.dropna().drop_duplicates(subset="gameId")

# colunas mais ligadas ao resultado
colunas_blue_side = ["blueGoldDiff", "blueExperienceDiff", "blueKills",
                     "blueDeaths", "blueDragons"]

atributos_partidas = partidas[colunas_blue_side]
vitorias_blue_side = partidas["blueWins"]

print("Partidas:", len(partidas))
print(vitorias_blue_side.value_counts(normalize=True).round(3).to_string(), "\n")


# KNN usa distancia: precisa padronizar
# Pipeline refaz a padronizacao a cada fold
def modelo_knn(k):
    return Pipeline([("escala", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=k))])


# estratificado mantem a proporcao das classes
validacao_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

valores_k = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101]

acuracias_knn = {}
print("  K | acuracia | desvio")
for k in valores_k:
    folds_knn = cross_val_score(modelo_knn(k), atributos_partidas, vitorias_blue_side,
                                cv=validacao_5fold, scoring="accuracy")
    acuracias_knn[k] = folds_knn.mean()
    print(f"{k:>3} |   {folds_knn.mean():.4f} | {folds_knn.std():.4f}")

melhor_k = max(acuracias_knn, key=acuracias_knn.get)
pior_k = min(acuracias_knn, key=acuracias_knn.get)

print(f"\nMelhor: K = {melhor_k} ({acuracias_knn[melhor_k]:.4f})")
print(f"Pior:   K = {pior_k} ({acuracias_knn[pior_k]:.4f})")
print(f"Diferenca: {(acuracias_knn[melhor_k] - acuracias_knn[pior_k]) * 100:.2f} pontos percentuais")


# NB nao usa distancia: dispensa padronizacao
folds_nb = cross_val_score(GaussianNB(), atributos_partidas, vitorias_blue_side,
                           cv=validacao_5fold, scoring="accuracy")
acuracia_nb = folds_nb.mean()
print(f"\nNaive Bayes: {acuracia_nb:.4f} | desvio {folds_nb.std():.4f}")

# comparacao: melhor K do KNN contra o NB
if acuracias_knn[melhor_k] > acuracia_nb:
    print(f"Melhor modelo para o dataset: KNN (K = {melhor_k})")
else:
    print("Melhor modelo para o dataset: Naive Bayes")
print(f"Diferenca: {abs(acuracias_knn[melhor_k] - acuracia_nb) * 100:.2f} pontos percentuais")