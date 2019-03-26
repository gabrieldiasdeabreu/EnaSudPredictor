'''
experimentos com rede neural alimentada por grupos separados e redes diferentes
'''
from EntradaRna import EntradaRna
from SerieTemporal import SerieTemporal
from Rna import Rna
# import sys
# import numpy as np
from keras import backend as K
# from keras.models import model_from_json
# import numpy as np
# import gc
serieTemporal = SerieTemporal('Instancias/EnaSudNovoAte2016.csv')
# tamanhoConjuntoTeste = 5
anoFinal = len(serieTemporal.serieDesnormalizada)-13
print(anoFinal)
# numEpocas = 1000
# ordemEntrada = 12
# ordensEntrada = range(1,37)#37
# mes = int(sys.argv[1])#outubro
# gruposDeMeses = [[1], [2], [3],[4],[5], [6], [7], [8], [9], [10], [11], [12]]
treino = [i for i in range(1, 13)]

# ordem, qtdNeuronios , epocas; , (31, 29, 2000, 2), (23,22, 80, 3), (21,25, 2000, 4 ) , (33,16, 2000, 8), (15, 45 , 500, 9), (31,3 , 10, 10), (32,45, 150, 11),
listaMelhores = [
                (29, 1, 20, 1),
                (31, 29, 4, 2),
                (23, 22, 40, 3),
                (21, 25, 9, 4),
                (30, 4, 8, 5),
                (23, 3, 4, 6),
                (22, 2, 2, 7),
                (33, 16, 9, 8),
                (15, 45, 100, 9),
                (31, 3, 4, 10),
                (32, 45, 11, 11),
                (24, 1, 8, 12)
                ]
incerto = max([x[3] for x in listaMelhores])

buffer = [None]*12
bufferHistTreino = [None]*12
bufferHistVal = [None]*12
melhoresModelos = [0]*12


def salvaModelo(modelo, mes):
    # serialize model to JSON
    model_json = modelo.to_json()
    with open("modelo/modelo_"+str(mes)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    modelo.save_weights("modelo/model_"+str(mes)+".h5")
    # print("Saved model to disk")

# mes = 1

def escreveTreino(bufferHistTreino, nomeSalvar):
    with open('saida/' + nomeSalvar, 'w') as arq:
        arq.write('ordem,qtdNeuronios')
        for i in range(1, incerto+1):
            arq.write(','+str(i))
        arq.write('\n')
        # print(bufferHistTreino)
        for histTreino in bufferHistTreino:
            if histTreino is not None:
                arq.write(str(histTreino[0]) + ',' + str(histTreino[1]))
                for epoca in histTreino[2]:
                    arq.write(',' + str(epoca))
                arq.write('\n')


for ordem, qtdNeuronios, numEpocas, mes in listaMelhores:    
    print('ordem:', ordem, 'Mes:', mes, 'qtdNeuronios:', qtdNeuronios, 'numEpocas:', numEpocas, 'treino:', treino)
    # entrada que a rede neural ira usar
    entrada = EntradaRna(serieTemporal, anoFinal, ordem)
    # prepara o treino utilizado para o mes
    entrada.preparaTreinoComListaMesesEspecificos(anoFinal, [mes])
    # entrada.preparaTesteComMesEspecifico(anoFinal, mes)
    # entrada.salvaTreino()

    # bufferTreino[mes-1] += entrada.salvaTreino()

    # print(ordem)
    # print(entrada.xTeste)
    # print(qtdNeuronios, sys.argv[1] , end=',', flush=True)
    rna = Rna(entrada)

    # usando todo o arquivo para calcular mape aqui bug
    rna.entradaRna.preparaTesteComMesEspecifico(anoFinal, mes)

    # rna.entradaRna.preparaTesteComMesEspecifico(anoFinal, mes)

    mapeMedioMelhorModelo = 10000000
    print('mes:', mes)
    for i in range(5000):
        listaCamadas = [(qtdNeuronios, 'relu'), (1, 'linear')]
        arquiteturaRna = rna.ArquiteturaRna(listaCamadas, verbose=0, epocas=numEpocas, optimizer='adam')
        # print
        # bufferHistTreino.append(rna.ArquiteturaRna(listaCamadas, verbose=0, epocas=numEpocas, optimizer='adam'))
        previsoes, mapeAtual = rna.previsaoSerie(anoInicial=anoFinal)
        # print(mapeAtual)
        # mapeAtual = rna.rna.evaluate(np.ndarray(entrada.xTeste), np.ndarray(entrada.yTeste))
        # mapeAtual = entrada.calculaMape(previsoes, entrada.yTeste)[2]
        # f = entrada.serieTemporal.desnormalizaElemento
        # print( f(previsoes), f(entrada.yTeste) )
        # print(entrada.serieTemporal.desnormalizaElemento(entrada.yTeste[-tamanhoConjuntoTeste:]))
        # print(i,'mapeTeste:', mapeAtual, 'mapeVal:', arquiteturaRna[-1])
        if mapeMedioMelhorModelo > mapeAtual:
            print(i, 'mapeTeste:', mapeAtual, 'mapeVal:', arquiteturaRna[0][-1], 'mapeTreino:', arquiteturaRna[1][-1])
            salvaModelo(rna.rna, mes)
            mapeMedioMelhorModelo = mapeAtual
            # melhoresModelos[mes-1] = rna.rna
            bufferHistTreino[mes-1] = ordem, qtdNeuronios, arquiteturaRna[1]
            bufferHistVal[mes-1] = ordem, qtdNeuronios, arquiteturaRna[0]
            buffer[mes-1] = entrada.escrevePrevisoes(previsoes, qtdNeuronios, mape=mapeMedioMelhorModelo)
            print('salvei')
        K.clear_session()
    # np.min(listaComExecucoes[1])
    # mes+=1
        # del(rna.rna)
        # del(rna.entradaRna)
        # del(rna)

# for melhorModelo, mes  in zip(melhoresModelos, range(1,13)):


escreveTreino(bufferHistTreino, 'saidaTreinoMape.csv')
escreveTreino(bufferHistVal, 'saidaVal.csv')

# for i in range(1, 13):
for coisa in buffer:
    if coisa is not None:
        with open('saida/testeDaRNA'+'mes_'+str(buffer.index(coisa)+1)+'.csv', 'w') as arq:
            arq.write(str(coisa))

