from keras.models import model_from_json
from EntradaRna import EntradaRna
from SerieTemporal import SerieTemporal

def leModelo(mes):
    # Model reconstruction from JSON file
    with open('melhoresModelos/modelo_'+str(mes)+'.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights('melhoresModelos/model_'+str(mes)+'.h5')
    model.compile('adam', 'mean_absolute_percentage_error')
    # Sequential.predict()
    return model


def salvaPredicao(real, previsto, mes, mapeMes):
    with open('saidaModeloPraCenarios_'+str(mes)+'.csv', 'w') as arquivo:
        arquivo.write('previsto,real,diferenca')
        arquivo.write(',mapeMes=' + str(mapeMes)+'\n')
        for i in zip(real, previsto):
            # print(i[1][0])
            previsaoDesn = SerieTemporal.desnormalizaElemento(i[1][0])
            realDesn = SerieTemporal.desnormalizaElemento(i[0])
            dif = previsaoDesn - realDesn
            arquivo.write(str(previsaoDesn)+','+str(real)+','+str(dif)+'\n')


def preparaModelo(SerieTemporal, DeOndePrever, ordemModelo, mes):
    '''
    Le modelo, executa, escreve saida e avalia o Mape
    '''

    entrada = EntradaRna(SerieTemporal, DeOndePrever, ordemModelo)
    # entrada.preparaTreinoComListaMesesEspecificos(DeOndePrever, [mes])
    entrada.preparaTesteComMesEspecifico(DeOndePrever, mes)
    modelo = leModelo(mes)
    previsto = modelo.predict(entrada.xTeste)
    mapeMes = modelo.evaluate(entrada.xTeste, entrada.yTeste)
    salvaPredicao(entrada.yTeste, previsto, mes, mapeMes)
    print(mapeMes)
