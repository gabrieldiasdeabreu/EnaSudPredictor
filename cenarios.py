from builtins import list

import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import glob
# size = 30000

def leBase(arquivo):
    db = pd.read_csv(arquivo)
    db = db['diferenca']
    # print(db)
    return db


def leBaseDados(arquivo):
    #gera aleatorio
    if arquivo is None:
        valoresAleatorios = scipy.stats.norm.rvs(213,123,size=size)
#         print(valoresAleatorios)
#         plt.plot(x, valoresAleatorios)
#         plt.show()
        #scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5,size=size)*47))
    else:
        valoresAleatorios = leBase(arquivo)
    #histograma # h = plt.hist(y, bins=range(48)) # randomValues
    return valoresAleatorios # print('nada') #distribuicoes usadas

def executaDistibuicoes(rvs):
    '''
    Procura em todas as distribuicoes
    '''
    #[ 'triang', 'laplace', 'logistic' ,'gamma' ,'beta', 'rayleigh', 'norm', 'pareto']#
    dist_names = [ 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford' , 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'] # dist_names = [ 'gamma' ,'betta', 'rayleigh', 'norm', 'pareto']#[ 'gamma' ,'beta', 'rayleigh', 'norm', 'pareto']#[ 'gamma' ,'betta', 'rayleigh', 'norm', 'pareto'][ 'alpha', 'beta', 'norm']# 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford' , 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'] # dist_names = [ 'gamma' ,'betta', 'rayleigh', 'norm', 'pareto']
    listaDistribuicoesResultado = list()
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        # print(y)
        param = dist.fit(rvs) # pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size # print(dist_name)
        # print(param)
        ks = scipy.stats.kstest(rvs, dist_name, param)
        listaDistribuicoesResultado.append((dist(*param), ks[1], param))
        # print( dist_name , ':' , ks[0])
        # plt.plot(pdf_fitted, label=dist_name)
        # plt.xlim(0,47)
    return listaDistribuicoesResultado # plt.legend(loc='upper right') # plt.show()

def retornaMelhorDistribuicao(rvs):
    '''
    Seleciona o melhor modelo de distribuicao
    '''
    resultados = executaDistibuicoes(rvs)
    # print('RESULTADOOOSS',resultados)
    # KSs = np.transpose(resultados)[1]
    melhorTupla = -1
    # menorKS = 10
    maiorP = -1
    for resultado in resultados:
        # if menorKS > resultado[1]:
            # menorKS = resultado[1]
        if maiorP < resultado[1]:
            maiorP = resultado[1]
            melhorTupla = resultado
    # print(melhorTupla)
    # return melhorTupla[0], menorKS, melhorTupla[2]
    return melhorTupla[0], maiorP, melhorTupla[2]
#     menor = np.min(KSs)
#     return resultados.index(menor)

#     np.min(np.transpose(resultados[1]))

def geraCenarios(distribuicao, numeroCenarios):
#     print(distribuicao)
    # x = scipy.arange(numeroCenarios)
    cenarios = distribuicao.rvs(size=numeroCenarios)
    # plt.hist(cenarios)
    # plt.show()
#     print(scipy.mean(cenarios))
#     print(distribuicao[0].stats(moments='m'))
    return cenarios


# def executaTodosModelos(diretorioBases):
#     '''
#         le de algum lugar as 12 bases para gerar o cenario
#     '''
#     arquivos = glob.glob( diretorioBases + 'diferencaPrevisao-real' )
# #     for arquivo in arquivos:
#     for arquivo in range(2):
#         # mudar aleatorio na de verdade
#         rvs = leBaseDados(arquivo, ehAleatorio=1)
#         distribuicao, ks, parametros = retornaMelhorDistribuicao(rvs)
#         print(distribuicao)
# #         print(distribuicao[0].stats(distribuicao[2], moments='m'))#(distribuicao[2] ))
#         cenarios = geraCenarios(distribuicao[0], 5000)
#         print(cenarios)
# #     return cenarios
# #         print('arquivo', )

def escreveSolucao(ondeEscrever, cenarios, distribuicao):
    with open(ondeEscrever, 'w') as arq:
        arq.write('cenarios, distribuicao: ,')
        arq.write(distribuicao.dist.name+',')
        arq.write(str(distribuicao.dist.shapes)+',')
        arq.write('loc,scale,')
        for arg in distribuicao.args:
            arq.write(str(arg)+',')
        arq.write('\n')
        for cenario in cenarios:
            arq.write(str(cenario)+'\n')


def geraCenariosComMelhorDistribuicaoEscreve(rvs, numeroCenarios, ondeEscrever):
    '''
    Gera cenarios com melhor distribuicao retorna cenarios e a melhor distribuicao
    '''
    distribuicao, pValue, parametros = retornaMelhorDistribuicao(rvs)
    # print(distribuicao.dist)
    cenarios = geraCenarios(distribuicao, numeroCenarios)
    escreveSolucao(ondeEscrever, cenarios, distribuicao)
    return distribuicao, pValue, cenarios
    # return geraCenarios(distribuicao, numeroCenarios), distribuicao.dist.name, distribuicao.dist.shapes, distribuicao.args

# def plotaPvalue(listaPvalues):
#
#     # valoresHist = listaMelhoresDist
#     # valoresHist = np.transpose(listaMelhoresDist)[2]
#     #arrumar aquiii
#
#     # pValues = [calculaPValue(valoresHist, real)  for real in      ]
#     print(valoresHist)
#     x = np.arange(1, 61)
#     y = [0.05]*60
#     plt.bar(x, valoresHist)
#     plt.plot(np.arange(14), y, color='red')
#     plt.title('P-Values por cenarios de cada mes')
#     plt.xlabel('meses')
#     plt.ylabel('P-Value')
#     # plt.xticks(x)
#     # plt.show()
#     plt.savefig('cenarios/execucaoCenario/PvaluesPorCenario.pdf')
#     plt.close()


# calculaPValue = lambda cenario, real: scipy.stats.ks_2samp(real, cenario)[1]

def fazCenarios(numCenarios):
    listaMelhoresDist = list()
    listaPvalues = list()
    for mes in range(1, 13):
        mes = str(mes)
        y = leBaseDados('cenarios/saidaModeloPraCenarios_'+mes+'.csv')
        listaMelhoresDist.append(geraCenariosComMelhorDistribuicaoEscreve(y, numCenarios, 'cenarios/cenarioGerado_'+mes+'.csv'))
#         valoresHist = np.transpose(listaMelhoresDist)[2]
#         print(valoresHist)
#         valoresHist = np.array_split(valoresHist, 5)#), (60,)) np.reshape(np.transpose(
#         print(valoresHist)
#     plotaPvalue(listaMelhoresDist)
#         geraTabelaMelhoresDistribuicoes()
# print(retornaMelhorDistribuicao(y))
# for resultado in resultados:
#     print(resultado)
