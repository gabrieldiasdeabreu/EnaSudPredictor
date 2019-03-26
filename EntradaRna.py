#classe para preparar a rede neural tanto a sua alimentacao quando saida
import numpy as np
from SerieTemporal import SerieTemporal
class EntradaRna:
    Serie = None
    serieTemporal = None
    xTreino = None
    yTreino = None
    xTeste = None
    yTeste = None
    ordemEntrada = 0

    def __init__(self, serieTemporal, anoFinalTreino, ordemEntrada):
        #inicia classe de Entranda que prepara a alimentacao da rede neural
        self.Serie = serieTemporal.Serie
        self.serieTemporal = serieTemporal
        # self.xTreino = list()
        # self.yTreino = list()
        # self.xTeste = list()
        # self.yTeste = list()
        # self.preparaTreino(anoFinalTreino, ordemEntrada)
        # self.preparaTeste(anoFinalTreino, ordemEntrada)
        self.ordemEntrada = ordemEntrada



    def preparaTreinoComMesEspecifico(self, anoFinalTreino, mesEspecifico):

        # print('Anos Totais :', len(self.Serie))
        # print('Anos no Treino :', len(self.Serie[:anoFinalTreino][:]))
        xAux = list(self.Serie[:anoFinalTreino, mesEspecifico-1].ravel())
        # for i in xAux:
        #     print(self.serieTemporal.desnormalizaElemento(i))
        while len(xAux) > self.ordemEntrada:
            self.xTreino.append(np.array(xAux[:self.ordemEntrada]))
            self.yTreino.append(xAux[self.ordemEntrada])
            # print(xAux[:ordemEntrada], xAux[ordemEntrada])
            xAux.pop(0)
        self.xTreino = np.array(self.xTreino)
        self.yTreino = np.array(self.yTreino)
        # print(self.xTreino)

    def preparaTreinoComListaMesesEspecificos(self, anoFinalTreino, mesEspecifico):
        #recebe ate que ano e os meses para gerar a alimentacao da rede neural
        # print('Anos Totais :', len(self.Serie))
        # print('Anos no Treino :', len(self.Serie[:anoFinalTreino][:]))
        # xAux = list(self.Serie[:anoFinalTreino, mesEspecifico-1].ravel())
        self.xTreino = list()
        self.yTreino = list()
        xAux = list()
        for anos in self.Serie[:anoFinalTreino] :
            for mes in mesEspecifico:
                 xAux.append(anos[mes-1])
        # return 2
        # for i in xAux:
        #     print(self.serieTemporal.desnormalizaElemento(i))
        while len(xAux) > self.ordemEntrada:
            self.xTreino.append(np.array(xAux[:self.ordemEntrada]))
            self.yTreino.append(xAux[self.ordemEntrada])
            # print(xAux[:ordemEntrada], xAux[ordemEntrada])
            xAux.pop(0)
        self.xTreino = np.array(self.xTreino)
        self.yTreino = np.array(self.yTreino)
        # print(self.xTreino)




    def preparaTesteComMesEspecifico(self, anoFinalTreino, mesEspecifico, ateOnde=-1):
        # prepara o treino para o mes e a partir do ano
        # print('Anos no Teste :', len(self.Serie[anoFinalTreino:]))
        self.xTeste = list()
        self.yTeste = list()
        if ateOnde is -1:
            xAux = list(self.Serie[anoFinalTreino - self.ordemEntrada: , mesEspecifico-1].ravel())
        else:
            xAux = list(self.Serie[anoFinalTreino - self.ordemEntrada: ateOnde, mesEspecifico-1].ravel())

        while len(xAux) > self.ordemEntrada:
            self.xTeste.append(np.array(xAux[:self.ordemEntrada]))
                               # .reshape(1,self.ordemEntrada))
            self.yTeste.append(xAux[self.ordemEntrada])
            # print(xAux[:ordemEntrada], xAux[ordemEntrada])
            xAux.pop(0)
        self.xTeste = np.array(self.xTeste)
        self.yTeste = np.array(self.yTeste)

    def salvaTreino(self):
        #gera um buffer com o treino realizado pela rna
        buffer = ''
        # with open('treinoDaRNA.csv', 'a') as arq:
        for i in range(self.ordemEntrada):
            buffer += 'neuronioEnt_'+str(i+1)+','
        buffer += 'resposta'+'\n'

        for x, y  in zip(self.xTreino, self.yTreino):
            for num in x:
                buffer +=str(self.serieTemporal.desnormalizaElemento(num))+','
            buffer += str(self.serieTemporal.desnormalizaElemento(y))+'\n'
        return buffer

    def escrevePrevisoes(self, previsoes, numNeuroniosCamadaOculta = ' ', mape=-1):
        #gera um buffer com as previsoes realizado pela rna
        # if mape == -1:
            # listaMapes = self.calculaMape(previsoes, self.yTeste)
        # with open('testeDaRNA.csv', 'a') as arq:
        buffer = 'MapeMedio='+str(mape)
        # buffer += 'numNeuroniosCamadaOculta='+ str(numNeuroniosCamadaOculta)+','+'ordemRede='+ str(self.ordemEntrada)+','+'melhorMape='+ str(listaMapes[1])+','++','+'desvioPadraoMapeMedio='+ str(listaMapes[3])+'\n'
        # for i in range(self.ordemEntrada):
        #     buffer += 'neuronioEnt_'+str(i+1)+','
        # buffer += 'previsao'+','+'real'+','+'Mape'+'\n'
        #
        # for x, previsao, y, mape in zip(self.xTeste, previsoes, self.yTeste, listaMapes[0]):
        #     for xaux in x:
        #         buffer += str(self.serieTemporal.desnormalizaElemento(xaux))+','
        #     buffer+=str(self.serieTemporal.desnormalizaElemento(previsao[0]))+','+str(self.serieTemporal.desnormalizaElemento(y))+','+ str(mape) +'\n'
        return buffer



    def calculaMape(self, previsoes, reais):
        #retorna o mape das previsoes com minimo media e desvio padrao
        listaMapes = [np.abs((self.serieTemporal.desnormalizaElemento(previsoes[i][0]) - self.serieTemporal.desnormalizaElemento(reais[i]))/self.serieTemporal.desnormalizaElemento(reais[i]))*100 for i in range(len(previsoes))]
        return listaMapes, min(listaMapes), np.mean(listaMapes), np.std(listaMapes)


#arrumar depois
#     def preparaTreino(self, anoFinalTreino):
#         #funcao com defeito nao usar
#         print('Anos Totais :', len(self.Serie))
#         print('Anos no Treino :', len(self.Serie[:anoFinalTreino][:]))
#         xAux = list(self.Serie[:anoFinalTreino][:].ravel())
#         while len(xAux) > self.ordemEntrada:
#             self.xTreino.append(np.array(xAux[:self.ordemEntrada]))
#             self.yTreino.append(xAux[self.ordemEntrada])
#             # print(xAux[:ordemEntrada], xAux[ordemEntrada])
#             xAux.pop(0)
#         self.xTreino = np.ndarray(xTreino)
#         print(self.xTreino)
# #arrumar depois
#     def preparaTeste(self, anoFinalTreino):
#         #com defeito nao usar
#         print('Anos no Teste :', len(self.Serie[anoFinalTreino:][:]))
#         xAux = list(self.Serie[anoFinalTreino:][:].ravel())
#         while len(xAux) > self.ordemEntrada:
#             self.xTreino.append(xAux[:self.ordemEntrada])
#             self.yTreino.append(xAux[self.ordemEntrada])
#             print(xAux[:self.ordemEntrada], xAux[self.ordemEntrada])
#             xAux.pop(0)

# def preparaTesteComListaMesesEspecificos(self, anoFinalTreino, mesEspecifico):
#     # print('Anos no Teste :', len(self.Serie[anoFinalTreino:]))
#     # xAux = list(self.Serie[anoFinalTreino - self.ordemEntrada:, mesEspecifico-1].ravel())
#     xAux = list()
#     for anos in self.Serie[anoFinalTreino-self.ordemEntrada:]:
#         for mes in mesEspecifico:
#              xAux.append(anos[mes-1])
#     while len(xAux) > self.ordemEntrada:
#         self.xTeste.append(np.array(xAux[:self.ordemEntrada]))
#                            # .reshape(1,self.ordemEntrada))
#         self.yTeste.append(xAux[self.ordemEntrada])
#         # print(xAux[:ordemEntrada], xAux[ordemEntrada])
#         xAux.pop(0)
#     self.xTeste = np.array(self.xTeste)
#     self.yTeste = np.array(self.yTeste)
#     print(self.serieTemporal.desnormalizaElemento(self.xTeste))
#     print(self.serieTemporal.desnormalizaElemento(self.yTeste))
