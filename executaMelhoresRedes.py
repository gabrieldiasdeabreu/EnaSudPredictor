from keras.models import model_from_json, load_model
from EntradaRna import EntradaRna
from SerieTemporal import SerieTemporal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from cenarios import fazCenarios
# from keras import Sequential


serie = SerieTemporal('Instancias/ENASud.csv')
ANO_FINAL_TREINO = len(serie.serieDesnormalizada)-10
#
#
# # print(modelo.evaluate(entrada.xTeste, entrada.yTeste))
# # print(entrada.xTeste)
#
# listaModelos = [(SerieTemporal, ANO_FINAL_TREINO, 29, 1)]
# [(SerieTemporal, ANO_FINAL_TREINO, ordemEntrada, mes)]


def leModelo( mes, diretorio='melhoresModelos'):
    # Model reconstruction from JSON file
    # with open(diretorio+'/modelo_'+str(mes)+'.json', 'r') as f:
    # model = model_from_json(f.read())
    # # Load weights into the new model
    # model.load_weights(diretorio+'/model_'+str(mes)+'.h5')
    # model.compile('adam', 'mean_absolute_percentage_error')
    #novo tipo de modelo
    model = load_model(diretorio+'/modeloMes=['+str(mes)+'].hdf5')
    # Sequential.predict()
    return model


def salvaPredicao(real, previsto, mes, mapeMes, nomeDir='cenarios/'):
    with open(nomeDir+'saidaModeloPraCenarios_'+str(mes)+'.csv', 'w') as arquivo:
        arquivo.write('previsto,real,diferenca')
        arquivo.write(',mapeMes=' + str(mapeMes)+'\n')
        for i in zip(real, previsto):
            # print(i)
            previsaoDesn = serie.desnormalizaElemento(i[1][0])
            realDesn = serie.desnormalizaElemento(i[0])
            dif = previsaoDesn - realDesn
            arquivo.write(str(previsaoDesn)+','+str(realDesn)+','+str(dif)+'\n')


def preparaModelo(serieTemporal, mes, nomeDir='cenarios/', ateOnde=-1):
    '''
    le o modelo prepara o teste e executa, salvando sua diferenca, previsao e real
    de um inicio ate um fim
    '''
    modelo = leModelo(mes)
    ordemModelo = modelo.input_shape[1]
    entrada = EntradaRna(serieTemporal, ordemModelo, ordemModelo)
    # entrada = EntradaRna(SerieTemporal, DeOndePrever, ordemModelo)
    # entrada.preparaTreinoComListaMesesEspecificos(DeOndePrever, [mes])
    # print(entrada.Serie)
    entrada.preparaTesteComMesEspecifico(ordemModelo, mes, ateOnde=ateOnde)
    # print('teste e treino', entrada.xTeste, entrada.yTeste)
    previsto = modelo.predict(entrada.xTeste, verbose=0 )
    mapeMes = modelo.evaluate(entrada.xTeste, entrada.yTeste, verbose=0)
    # print('aqui',entrada.yTeste, 'aqui')
    salvaPredicao(entrada.yTeste, previsto, mes, mapeMes, nomeDir)
    print('mape', mapeMes)
    print('ordem', ordemModelo)

def executaMelhoresRedes():
    '''
    procedimento para selecionar e gerar a diferenca dos melhores modelos
    '''
    for mes in range(1,13):        
        print(mes)
        preparaModelo(serie,  mes, ateOnde=ANO_FINAL_TREINO)    

def criaListaCenariosJuntos():
    '''
    junta series de cenarios como se fosse uma serieTemporal
    '''
    listaCenarios = []
    ultimosValoresReais = achaUltimosValoresTreinoSerie(SerieTemporal)
    for mes in range(1,13):
        listaCenarios.append(pd.read_csv('cenarios/cenarioGerado_'+str(mes)+'.csv')['cenarios'])

    with open('cenarios/cenariosGeradosTodosJuntos.csv', 'w') as arq:
        for cenariosMeses in zip(*listaCenarios):
            erro = False
            for  aux in [dif+real for dif, real in zip(cenariosMeses, ultimosValoresReais)]:
                if aux < 0:
                    erro = True
            if not erro:
                linha = ','.join([str(dif+real) for dif, real in zip(cenariosMeses, ultimosValoresReais)])
                arq.write(linha+'\n')


# executaMelhoresRedes()
def executaModeloNoCenario(mes):
    global serie
    serieCenarios = SerieTemporal('cenarios/cenariosGeradosTodosJuntos.csv', serie.menorElemento, serie.maiorElemento)
    print(mes)
    preparaModelo(serieCenarios, mes, 'cenarios/execucaoCenario/')


def plotaResultado(real, cenarios):
    #muda para 5 anos
    real = real*5
    # print(cenarios)
    cenarios = np.reshape(np.transpose([np.array_split(x, 5) for x in cenarios]), (60, ))

    previsao = [np.mean(x) for x in cenarios]
    print(cenarios)
    x = np.arange(1, 61) #13
    for cenario in zip(*cenarios):
        # print('cenario', '-'*1000)
        # print(cenario)
        plt.plot(x, cenario, c='black', linestyle='-.', linewidth=0.1)
    plt.plot(x, real, c='red', label='real', marker='.')
    plt.plot(x, previsao, c='blue', label='previsao', marker='.')
    plt.xticks(x)
    plt.title('cenarios Afluencia EnaSud')
    plt.xlabel('meses')
    plt.ylabel('Vazao de Afluência')
    plt.legend()
    plt.savefig('cenarios/execucaoCenario/MediaCenariosMediaMeses2.pdf')
    plt.close()


def executaMelhoresModelosNosCenarios(numAnos):
    global serie
    # achaUltimosValoresSerie(serie)
    # previsoesMedias = [0]*12
    listaCenarios = [None]*12
    for mes in range(1,13):
        # cenarios = pd.read_csv('cenarios/cenarioGerado_'+str(mes)+'.csv')['cenarios']
        # ultimoValorReal = serie.serieDesnormalizada[-1][mes-1]
        # print(cenarios + ultimoValorReal)
        executaModeloNoCenario(mes)
        cenarios = pd.read_csv('cenarios/execucaoCenario/saidaModeloPraCenarios_'+str(mes)+'.csv')['previsto']        
        listaCenarios[mes-1] = cenarios

    # só do teste
    realMedias = [np.mean(x) for x in np.transpose(serie.serieDesnormalizada[ANO_FINAL_TREINO:])]    
    # print(realMedias)
    plotaResultado(realMedias,  listaCenarios)



def achaUltimosValoresTreinoSerie(SerieTemporal):
    aux = [serie.serieDesnormalizada[ANO_FINAL_TREINO-1][mes-1] for mes in range(1,13)]
    return aux

def plotaPvalues():
    #usando treino
    realTodosMeses = np.transpose(serie.serieDesnormalizada[ANO_FINAL_TREINO:])
    cenarios = np.loadtxt('cenarios/cenariosGeradosTodosJuntos.csv', delimiter=',')
    listaPValues = list()
    for mes in range(1, 13):
        cenario = cenarios[:, mes-1]
        # print(cenario)
        #divide em cinco
        valoresHist = np.array_split(cenario, 5)
        # print(valoresHist)
        real = realTodosMeses[mes-1]
        # print('real', real)
        for i in valoresHist:
            print(i)
        aux = [scipy.stats.ks_2samp(val, real) for val in valoresHist]
        for i in aux:
            print('PValue', mes, i[1])
            listaPValues.append(i[1])
    print(listaPValues)
    plotaPValues(listaPValues)

def plotaPValues( pValues):
    x = np.arange(1, 61)
    y = [0.05] * 60
    plt.bar(x, pValues)
    plt.plot(np.arange(60), y, color='red')
    plt.title('P-Values por cenarios de cada mes')
    plt.xlabel('meses')
    plt.ylabel('P-Value')
    # plt.xticks(x)
    # plt.show()
    plt.savefig('cenarios/execucaoCenario/PvaluesPorCenario.pdf')
    plt.close()

print('gerando o modelo e diferencas', '-'*100)

executaMelhoresRedes()

print('gerando os cenarios', '-'*100)

fazCenarios(1000)
plotaPvalues()

print('executando e plotando Grafico---------------------------------------------------')

criaListaCenariosJuntos()
executaMelhoresModelosNosCenarios(5)
