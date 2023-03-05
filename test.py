import time
import matplotlib.pyplot as plt
import numpy
from statistics import mean
from arbolParidad import arbolParidad, arbolParidadAtGeom
import copy
from enum import Enum

class Test():    
    def __init__(self, k, n, l, caos = False):
        self.l, self.k, self.n = l, k, n
        self.caos = caos
        self.A, self.B = arbolParidad(k, n, l, caos), arbolParidad(k, n, l, caos)
        self.sincronizacion()
        
    def sincronizacion(self):
        self.porcentajeSincro = self.A.sincronizacionCon(self.B)
    
    def entradaAleatoria(self):
    	return numpy.random.randint(-self.l, self.l + 1, [self.k, self.n])
    
    def sincroMaxLista(self, Es):
        sincro = 0
        for E in Es:
            if E.sincronizacionCon(self.A) > sincro:
                sincro = E.sincronizacionCon(self.A)
        return sincro
    
    def coordinaSalidas(self):
        X = self.entradaAleatoria()   
        
        xA, xB = self.A(X), self.B(X) 
    
        self.A.propagacionHaciaAtras(xB)
        self.B.propagacionHaciaAtras(xA)     
        
        self.sincronizacion()        
        self.historial.append(self.porcentajeSincro)
        return X, xA, xB
        
     
    def testBasico(self):
        self.historial, cont, t_inicial = [], 0, time.time()
        
        while(self.A.hash_pesos() != self.B.hash_pesos()):
            self.coordinaSalidas()
            cont += 1
            
        return (0, time.time() - t_inicial, cont)
        
    def fuerzaBruta(self):
        self.historial, cont, t_inicial, E = [], 0, time.time(), arbolParidad(self.k, self.n, self.l, self.caos)  
        
        while(self.porcentajeSincro != 100):
            X, xA, xB = self.coordinaSalidas()            
            if xA == xB == E(X):
                E.propagacionHaciaAtras(xA)
            cont += 1
        
        return (E.sincronizacionCon(self.A), time.time() - t_inicial, cont)
    
    def ataqueGeometrico(self, N = 10):
        self.historial, cont, t_inicial, Es = [], 0, time.time(), []
        for i in range(N):
            Es.append(arbolParidadAtGeom(self.k, self.n, self.l, self.caos))   
        
        while(self.porcentajeSincro != 100):
            X, xA, xB = self.coordinaSalidas()            
            if xA == xB:
                for E in Es:
                    E(X)
                    E.propagacionHaciaAtras(xA)
            cont += 1
        
        return (self.sincroMaxLista(Es), time.time() - t_inicial, cont)
    
    def ataqueGenetico(self, N = 10, M = 200):
        self.historial, cont, t_inicial, Es = [], 0, time.time(), []
        for i in range(N):
            Es.append(arbolParidad(self.k, self.n, self.l, self.caos))  
            
        while(self.porcentajeSincro != 100):
            X, xA, xB = self.coordinaSalidas()
            if xA == xB:
                no1, EsCopia = int(xA == -1), list(Es)
                for E in EsCopia:
                    res = E(X)
                    if len(Es) + 2**(self.k-1) < M:
                        Es.remove(E)
                        for i in range(2**self.k):
                            aux, lista = i, []
                            for _ in range(self.k):
                                lista.append((-1)**int(aux%2))
                                aux = aux//2
                            if (lista.count(-1)%2 == no1):
                                Eaux  = copy.deepcopy(E)
                                Eaux.cambiaDatosIntermedios(lista)
                                Eaux.propagacionHaciaAtras(xA)
                                Es.append(Eaux)
                    elif xA == res:
                        E.propagacionHaciaAtras(xA) 
                    elif len(Es)>1: 
                        Es.remove(E)
            cont += 1
        
        return (self.sincroMaxLista(Es), time.time() - t_inicial, cont)
        
    def graficoSincronizacion(self):
        plt.plot(range(len(self.historial)), self.historial)
          
        plt.xlabel('Nº comunicacion')
        plt.ylabel('% sincronizacion')
        
        plt.show()
        
    class Ataques(Enum):
        FuerzaBruta = 'Ataque por Fuerza Bruta'
        Geometrico  = 'Ataque Geometrico'
        Genetico    = 'Ataque Genetico'  
     
def prueba(n: int, ataque: Test.Ataques, caos: bool)-> int:        
    resS, resT, resC = [], [], []
    for i in range(n): 
        t = Test(5, 10, 10, caos)
        if ataque == Test.Ataques.FuerzaBruta:
            (sincro, tiempo, cont) = t.fuerzaBruta()
        elif ataque == Test.Ataques.Geometrico:
            (sincro, tiempo, cont) = t.ataqueGeometrico()
        elif ataque == Test.Ataques.Genetico:
            (sincro, tiempo, cont) = t.ataqueGenetico()
        else:
            print('b')
            (sincro, tiempo, cont) = t.testBasico()
        resS.append(sincro)
        resT.append(tiempo)
        resC.append(cont)
        if sincro == 100:
            print('Roto por el atacante')
        
    plt.plot(range(len(resS)), resS)
    plt.xlabel('Nº iteración prueba')
    plt.ylabel('% sincronización red externa')    
    plt.show()
    
    plt.plot(range(len(resC)), resC, color="red")
    plt.xlabel('Nº iteración prueba')
    plt.ylabel('Nº de interacciones')    
    plt.show()
    
    print("La media del porcentaje de sincronizacion es: ", str(mean(resS)), " porciento")
    print("La media del numero de interacciones es: ", str(mean(resC)))
    print("La media del tiempo para sincronizarse es: ", str(mean(resT)), " segundos")
    return resS
     
#prueba(1, Test.Ataques.Geometrico, False)