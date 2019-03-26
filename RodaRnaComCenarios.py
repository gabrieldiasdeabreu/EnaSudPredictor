import cenarios
import ModeloRedeNeural as M
from EntradaRna import EntradaRna
from SerieTemporal import SerieTemporal

ANO_FINAL_TREINO = 75-11
serie = SerieTemporal('Instancias/ENASud.csv')
listaModelos = [1]

for modelo in listaModelos:
    print(modelo)
    modelo = M.leModelo(modelo)
    
