import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
tamanhoTeste = 11
NumMaximoNeuronios = 50
TamMaximoOrdem = 36

def leArquivo(mes):
    listaListaMapes = list()
    for dir in glob.glob('experimentos*/experimento_*/testeDaRNA_ex_*mes_'+str(mes)+'.csv'):
        print(dir)
        listaListaMapes.append(retornaListaMapes(dir))
    listaResultante, listaDesvios = calculaMediaMapes(listaListaMapes)
    return escreveArquivo(listaResultante, listaDesvios, mes)
    # plotaMapes(listaResultante)


def retornaListaMapes(dir):
    listaMapes = list()
    arquivo = open(dir,'r')
    i=0
    for linha in arquivo.readlines():
        if(i % (tamanhoTeste+2) == 0):#numero de instancias para teste mais 2
            # print(linha[str(linha).find('MapeMedio'):].split('=')[1].split(',')[0])
            listaMapes.append(float(linha[str(linha).find('MapeMedio'):].split('=')[1].split(',')[0]))
            # float(linha[str(linha).rfind('MapeMedio'):].split('=')[1].split('\n')[0]))
        i+=1
    return listaMapes

def calculaMediaMapes(listalistaMapes):
    listaResultante = list()
    listaDesvios = list()
    for i in range(len(listalistaMapes[0])):
        listaAux = list()
        for execucao in range(len(listalistaMapes)):
            listaAux.append(listalistaMapes[execucao][i])
        listaResultante.append(np.mean(listaAux))#, np.std(listaAux)))
        listaDesvios.append(np.std(listaAux))
    return listaResultante, listaDesvios

def escreveArquivo(listaResultante, listaDesvios, mes):
    arquivoEscrita = open('matrizMape_OrdemXNeuronios_mes'+str(mes)+'.csv','w')
    arquivoEscrita.write('Ordem/NumNeuroniosCamadaOculta')
    for i in range(1, NumMaximoNeuronios+1):
        arquivoEscrita.write(','+str(i))
    arquivoEscrita.write('\n')
    vezesEscreveu = 1
    for i in range(len(listaResultante)):
        if i % 50 == 0:
            arquivoEscrita.write(str(vezesEscreveu)+','+str(listaResultante[i])+'+-'+str(listaDesvios[i]))
        elif i % 50 == 49:
            arquivoEscrita.write(','+str(listaResultante[i])+'+-'+str(listaDesvios[i])+'\n')#+'+-'+str(listaResultante[i][1])+'\n')
            vezesEscreveu+=1
        else :
            arquivoEscrita.write(','+str(listaResultante[i])+'+-'+str(listaDesvios[i]))#+'+-'+str(listaResultante[i][1]))
    indMin = listaResultante.index(min(listaResultante))
    arquivoEscrita.write('melhorResultado='+str(listaResultante[indMin])+'+-'+str(listaDesvios[indMin]))
    return str(listaResultante[indMin])+','+str(listaDesvios[indMin])+','+str( int(indMin / 50)+1) +','+ str((indMin%50)+1)

def plotaMapes(listaResultante):
    num = 10
    lista = np.array(listaResultante).reshape(36, 50)
    x = np.arange(1,51)
    for i in range(len(lista)-25):
        plt.plot(x, lista[i], label=str(i+1))
    plt.xLabel='Numero de neuronios na camada oculta'
    plt.yLabel='Mape medio'
    plt.legend()
    plt.show()

def executaParaTodosOsMeses():
    mes = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun','jul', 'ago', 'set', 'out', 'nov', 'dez']
    listaSaidas = list()
    for i in range(1,13):
        listaSaidas.append(leArquivo(i))
    with open('listaMelhoresRnasDoExperimentos.csv', 'w') as arq:
        arq.write('mes,MAPES,STD,ordem,númeroNeurôniosCamadaOculta\n')
        for i in range(12):
            arq.write(mes[i]+','+listaSaidas[i]+'\n')

executaParaTodosOsMeses()
#leArquivo()
