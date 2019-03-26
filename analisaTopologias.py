'''gera os graficos das epocas de treino para cada grupo de numero de neuronios
na camada oculta
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def geraGraficos(df):
    x = np.arange(1,1001)
    mes= 0
    contador = 1
    for linhas in df.iterrows():
        plt.plot(x, linhas[1][2:], label='N' + str(linhas[1][1]))
        plt.legend()
        # plt.show()
        if linhas[1][1] == 50:
            plt.title('ordem: '+ str(linhas[1][0]) + ' mes:' + str((mes%12)+1))
            plt.xlabel('Epocas')
            plt.ylabel('Mape')
            plt.savefig('graficos/graficoMapeEpoca'+ 'Ordem_'+ str(linhas[1][0]) + ' Mes_' + str((mes%12)+1) + '.png')
            plt.close()
            mes+=1
            contador=1
        # print(str(linhas[1][0]), str(linhas[1][1]))
        contador+=1

df = pd.read_csv('historicoDoTreino_ex_0meses_[12].csv')
# print(df)
geraGraficos(df)
