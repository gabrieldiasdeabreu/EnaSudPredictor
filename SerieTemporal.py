import numpy as np
class SerieTemporal:
    Serie = None
    serieDesnormalizada = None
    menorElemento = 0
    maiorElemento = 0

    def __init__(self, nomeInstancia, min=0, max=0):
        '''lê um arquivo do tipo csv'''
        self.Serie = np.loadtxt(nomeInstancia, 'float', delimiter=',')
        self.serieDesnormalizada = self.Serie
        serieLinear = self.Serie.ravel()
        if min is 0 and max is 0:            
            self.menorElemento = np.min(serieLinear)
            self.maiorElemento = np.max(serieLinear)
        else:
            self.menorElemento = min
            self.maiorElemento = max
        self.normalizaSerie()

    def normalizaSerie(self):
        self.Serie = (self.Serie - self.menorElemento)/(self.maiorElemento-self.menorElemento)

    def desnormalizaElemento(self, elemento):
        return elemento*(self.maiorElemento-self.menorElemento) + self.menorElemento
