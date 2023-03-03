import time
import matplotlib.pyplot as plt
import numpy
from statistics import mean
from arbolParidad import arbolParidad, arbolParidadAtGeom
from copy import copy
from enum import Enum

class Test():    
    def __init__(self, k, n, l):
        self.l = l
        self.k = k
        self.n = n
        self.A = arbolParidad(k, n, l)
        self.B = arbolParidad(k, n, l)
        self.sincronizacion()
        
    def sincronizacion(self):
        self.porcentajeSincro = self.A.sincronizacionCon(self.B)
    
    def entradaAleatoria(self):
    	return numpy.random.randint(-self.l, self.l + 1, [self.k, self.n])
    
    def coordinaSalidas(self):
        X = self.entradaAleatoria()   
        
        xA = self.A(X) 
        xB = self.B(X) 
    
        self.A.propagacionHaciaAtras(xB)
        self.B.propagacionHaciaAtras(xA)     
        
        self.sincronizacion()        
        self.historial.append(self.porcentajeSincro)
        return X, xA, xB
        
     
    def testBasico(self, imprime = True):
        t_inicial = time.time() 
        self.historial = []
        cont = 0
        while(self.A.hash_pesos() != self.B.hash_pesos()):
            self.coordinaSalidas()
            cont += 1
        tiempo = time.time() - t_inicial
        if imprime:
            print ('Sincronizadas en ' + str(tiempo)+ " segundos")  
        return (tiempo, cont)
        
    def fuerzaBruta(self, imprime = True):
        E = arbolParidad(self.k, self.n, self.l)        
        t_inicial = time.time() 
        self.historial = []
        cont = 0
        while(self.porcentajeSincro != 100):
            X, xA, xB = self.coordinaSalidas()            
            if xA == xB == E(X):
                E.propagacionHaciaAtras(xA)
            cont += 1
        
        tiempo, sincro = time.time() - t_inicial, E.sincronizacionCon(self.A)
        if(imprime):
            print ('Sincronizadas en ' + str(tiempo)+ " segundos")    
            print("Maquina externa sincronizada un " + str(sincro) + " porciento ") 
        return (sincro, tiempo, cont)
    
    def ataqueGeometrico(self, imprime = True, N = 10):
        Es = []
        for i in range(N):
            Es.append(arbolParidadAtGeom(self.k, self.n, self.l))     
        t_inicial = time.time()
        self.historial = []
        cont = 0
        while(self.porcentajeSincro != 100):
            X, xA, xB = self.coordinaSalidas()            
            if xA == xB:
                for E in Es:
                    E(X)
                    E.propagacionHaciaAtras(xA)
            cont += 1
        
        tiempo, sincro = time.time() - t_inicial, 0
        for E in Es:
            if E.sincronizacionCon(self.A) >= sincro:
                sincro = E.sincronizacionCon(self.A)
        if(imprime):
            print ('Sincronizadas en ' + str(tiempo)+ " segundos")    
            print("La máquina externa más sincronizada, lo esta un " + str(sincro) + " porciento ") 
        return (sincro, tiempo, cont)
    
    def ataqueGenetico(self, imprime = True, N = 10, M = 200):
        Es = []
        for i in range(N):
            Es.append(arbolParidad(self.k, self.n, self.l))     
        t_inicial = time.time()
        self.historial = []
        cont = 0
        while(self.porcentajeSincro != 100):
            X, xA, xB = self.coordinaSalidas()
            if xA == xB:
                EsCopia = list(Es)
                for E in EsCopia:
                    if xA == E(X):
                        ldI = len(E.getDatosIntermedios())
                        if len(Es) + 2**ldI < M:
                            Es.remove(E)
                            for i in range(2**ldI):
                                aux = i
                                lista = []
                                for _ in range(ldI):
                                    lista.append((-1)**int(aux%2))
                                    aux = aux//2
                                Eaux  = copy(E)
                                Eaux.cambiaDatosIntermedios(lista)
                                Eaux.propagacionHaciaAtras(xA)
                                Es.append(Eaux)
                        else:
                            E.propagacionHaciaAtras(xA) 
                    elif len(Es)>1: 
                        Es.remove(E)
            cont += 1
        
        tiempo, sincro = time.time() - t_inicial, 0
        for E in Es:
            if E.sincronizacionCon(self.A) > sincro:
                sincro = E.sincronizacionCon(self.A)
        if(imprime):
            print ('Sincronizadas en ' + str(tiempo)+ " segundos")    
            print("La máquina externa más sincronizada, lo esta un " + str(sincro) + " porciento ") 
        return (sincro, tiempo, cont)
        
    def graficoSincronizacion(self):
        plt.plot(range(len(self.historial)), self.historial)
          
        plt.xlabel('nº comunicacion')
        plt.ylabel('porcentaje sincronizacion')
        
        plt.show()
        
    class Ataques(Enum):
        FuerzaBruta = 'Ataque por Fuerza Bruta'
        Geometrico  = 'Ataque Geométrico'
        Genetico    = 'Ataque Genético'  
     
def prueba(n: int, ataque: Test.Ataques)-> int:        
    resS, resT, resC = [], [], []
    for i in range(n):       
        t = Test(5, 10, 10)
        if ataque == Test.Ataques.FuerzaBruta:
            (sincro, tiempo, cont) = t.fuerzaBruta(False)
        elif ataque == Test.Ataques.Geometrico:
            (sincro, tiempo, cont) = t.ataqueGeometrico(False)
        elif ataque == Test.Ataques.Genetico:
            (sincro, tiempo, cont) = t.ataqueGenetico(False)
        else:
            sincro = 0
            (tiempo, cont) = t.testBasico(False)
        resS.append(sincro)
        resT.append(tiempo)
        resC.append(cont)
        
    plt.plot(range(len(resS)), resS)
    plt.xlabel('nº iteración prueba')
    plt.ylabel('% sincronizacion red externa')    
    plt.show()
    
    plt.plot(range(len(resC)), resC, color="red")
    plt.xlabel('nº iteración prueba')
    plt.ylabel('Nº de interacciones')    
    plt.show()
    
    print("La media del porcentaje de sincronización es: ", str(mean(resS)), "%")
    print("La media del número de interacciones es: ", str(mean(resC)))
    print("La media del tiempo para sincronizrse es: ", str(mean(resT)), "segundos")
    return resS
     
prueba(100, Test.Ataques.Geometrico)