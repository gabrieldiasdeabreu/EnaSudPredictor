import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

def geraGraficos(treino, val):

    for linhaTreino, linhaVal, mes in zip(treino.itertuples(), val.itertuples(), range(1,13)):
        treino = linhaTreino[3:]
        val = linhaVal[3:]
        x = range(1, len(treino)+1)
        # print()
        plt.plot(x, treino, color='blue', label='treino')
        plt.plot(x, val, color='red', label='validacao')
        plt.title('Mapes treino e validacão por época')
        plt.xlabel('Épocas')
        plt.ylabel('Mape')
        plt.legend()
        # pd.DataFrame(linha[1][2:]).plot.line()
    # plt.plot(range(1, len(df)))
        plt.savefig('graficos/GraficosTreino_' + str(mes))
        plt.close()

dir = 'saidaBackup/saida5/saida/'
Treino = pd.read_csv(dir+'saidaTreinoMape.csv', skiprows=0)
Validacao = pd.read_csv(dir +'saidaVal.csv', skiprows=0)
# print(df)
geraGraficos(Treino, Validacao)
