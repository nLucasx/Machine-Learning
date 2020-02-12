import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Flor():

    def __init__(self, tipo, distancia):
        self.tipo = tipo
        self.distancia = distancia

    def __lt__(self, nova_flor):
        return self.distancia < nova_flor.distancia


def conta_classes(df):
    virginica = versicolor = setosa = 0
    for i in range(0, df.shape[0]):
        if df.iloc[i][4] == 'virginica':
            virginica += 1
        elif df.iloc[i][4] == 'setosa':
            setosa += 1
        else:
            versicolor += 1
    return [virginica, setosa, versicolor]


def treino_e_teste(df):

    treino, teste = [], []
    virginica = versicolor = setosa = 0
    for i in range(0, df.shape[0]):
        if df.iloc[i][4] == 'virginica' and virginica < 30:
            treino.append(df.iloc[i])
            virginica += 1
        elif df.iloc[i][4] == 'setosa' and setosa < 30:
            treino.append(df.iloc[i])
            setosa += 1
        elif df.iloc[i][4] == 'versicolor' and versicolor < 30:
            treino.append(df.iloc[i])
            versicolor += 1
        else:
            teste.append(df.iloc[i])

    return treino, teste


def euclidiana(v1, v2):
    soma, dim = 0, len(v1)

    for i in range(0, dim-1):
        soma += np.square(int(v1[i]) - int(v2[i]))
    return np.sqrt(soma)


def knn(treinamento, amostra, k):

    dists = []
    tam = len(treinamento)

    for i in range(tam):
        distancia = euclidiana(amostra, treinamento[i])
        flor = Flor(treinamento[i][4], distancia)
        dists.append(flor)

    dists.sort()

    classes = [0, 0, 0]

    for i in range(k):
        if dists[i].tipo == 'virginica':
            classes[0] += 1
        elif dists[i].tipo == 'setosa':
            classes[1] += 1
        else:
            classes[2] += 1

    if classes[0] > classes[1] and classes[0] > classes[2]:
        return 'virginica'
    elif classes[1] > classes[0] and classes[1] > classes[2]:
        return 'setosa'
    elif classes[2] > classes[0] and classes[2] > classes[1]:
        return 'versicolor'


df = pd.read_csv(
    'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv')
treino, teste = treino_e_teste(df)
acertos = 0

for amostra in teste:
    classe = knn(treino, amostra, 7)
    if classe == amostra[4]:
        acertos += 1
print("Resultado:", acertos, "acertos de", 60, "\nCom k = 7\nAcurácia de ", 100*acertos/len(teste), "%")

print('\n\n\n\n\n\n\n\n')