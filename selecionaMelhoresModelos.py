""" Seleciona dentre os testes os melhores e
    copia para a pasta melhores modelos,
    gerando uma tabela de resultados no arquivo resultados
"""
# import pandas as pd
import glob as gl
import numpy as np
from subprocess import call



def montaTabela(mapes, dir):
    '''
    Escreve tabela em arquivo
    '''
    with open(dir+'resultados.csv', 'w') as arq2:
        arq2.write('mes' + ',' +'MAPE'+'\n')
        for mes in range(1,13):
            arq2.write(str(mes) + ',' + str(mapes[mes-1])+'\n')
    # tabela = []
    # for mes in range(1,13):
    #     tabela.append(mapes[mes-1])
    # return tabela

def leArquivo(dir):
    '''
    le arquivo do csv de teste
    '''
    mapes = [1000000]*12
    arquivos = [None]*12
    for nomeArquivo in gl.glob(dir+'testeDaRNAmes_*.csv'):
        with open(nomeArquivo) as arq:
            mes= nomeArquivo.split('_')[1].split('.')[0]
            # print(nomeArquivo, mes)
            mes = int(mes)-1
            cabecalho = arq.readline()
            mape = float(cabecalho[cabecalho.find('MapeMedio='):].split(',')[0].split('=')[1])
            mapes[mes] = mape
            # print(nomeArquivo)
            arquivos[mes] = nomeArquivo
            # print(mape, mes)
    return mapes, arquivos
    # return montaTabela(mapes, dir)


def copiaMelhorModelo(dir, mes):
    '''
    Copia melhor modelo para uma pasta de melhores
    '''
    # cmd1 = ['mkdir', 'melhoresModelos']
    dirModelos = dir.split('/')[:-2]
    dirModelos = '/'.join(dirModelos)
    print('modelos:', dirModelos)
    arquivoH5 = 'model_'+str(mes)+'.h5'
    arquivoJson = 'modelo_'+str(mes)+'.json'
    cmdCopiaH5 = ['cp', dirModelos+'/modelo/'+arquivoH5, './melhoresModelos/'+ arquivoH5]
    cmdCopiaJson = ['cp', dirModelos+'/modelo/'+arquivoJson, './melhoresModelos/'+ arquivoJson]
    print(cmdCopiaJson)
    print(cmdCopiaH5)
    call(cmdCopiaH5)
    call(cmdCopiaJson)
    # print(cmd1)
    # call(cmd)


def montaMelhorTabela(tabelas, diretorios):
    '''
    escolhe melhor das tabelas e copia os modelos para pasta
    '''
    tabelaResultante = []
    tabelas = np.transpose(tabelas)
    diretorios = np.transpose(diretorios)
    for i in range(1,13):
        # print(tabelas[i-1])
        indiceMenor = np.argmin(tabelas[i-1])
        melhorModeloDiretorio = diretorios[i-1][indiceMenor]
        # print('argmin:', indiceMenor,'valor', tabelas[i-1][indiceMenor],'esperado:',  np.min(tabelas[i-1]) )
        copiaMelhorModelo(melhorModeloDiretorio, i)
        tabelaResultante.append(np.min(tabelas[i-1]))
    montaTabela(tabelaResultante, './')

tabelas = []
diretorios = []
for dir in gl.glob('saidaBackup/saida*/saida/'):
    print('diretorio:',dir)
    mapes, nomeArquivo = leArquivo(dir)
    tabelas.append(mapes)
    diretorios.append(nomeArquivo)
# print(tabelas)
# print('dir:',diretorios[2])
call(['mkdir', 'melhoresModelos'])
montaMelhorTabela(tabelas, diretorios)
print('terminado :))')