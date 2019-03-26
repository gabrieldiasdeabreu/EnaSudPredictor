#classe para encapsular as tarefas da rede neural
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from EntradaRna import EntradaRna

class Rna:
    entradaRna = None
    mapeTreino = None
    rna = None

    def __init__(self, entradaRna):
        self.entradaRna = entradaRna

    def ArquiteturaRna(self, listaCamadas, verbose , epocas, optimizer='adam'):
        '''encapsula o metodo de criacao da rede neural e sua Arquitetura
        utilizando o keras recebendo uma lista de camadas e epocas e o otimizador'''
        self.rna = Sequential()
        # print(listaCamadas)
        self.rna.add(Dense(units=listaCamadas[0][0], input_dim=self.entradaRna.ordemEntrada,
                           activation=listaCamadas[0][1]))
        self.rna.add(Dropout(0.5))
        listaCamadas.pop(0)
        for i in listaCamadas:
            self.rna.add(Dense(units=i[0], activation=i[1]))
        self.rna.compile(optimizer=optimizer, loss='mean_absolute_percentage_error',
                         metrics=['mean_absolute_percentage_error'])
        # print('X:',self.L.x)
        # print('xTreino',len(self.entradaRna.xTreino))
        hist = self.rna.fit(self.entradaRna.xTreino, self.entradaRna.yTreino, 1,
                            validation_split=0.1095890410958904, verbose=verbose, epochs=epocas)
        # print(hist.history)
        return hist.history['val_loss'], hist.history['mean_absolute_percentage_error']

    def previsaoSerie(self, anoInicial = 2):
        '''retorna as previsoes atingidas e a metrica do teste'''
        return 1, self.rna.evaluate(self.entradaRna.xTeste, self.entradaRna.yTeste, verbose=0)[1]
        #self.rna.predict(self.entradaRna.xTeste, verbose=0),
        # self.entradaRna.escrevePrevisoes(previsoes)
