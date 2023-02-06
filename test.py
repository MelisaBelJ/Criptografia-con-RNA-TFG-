import time
import matplotlib.pyplot as plt
import numpy
from statistics import mean
from arbolParidad import arbolParidad

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
        
     
    def testBasico(self):
        t_inicial = time.time() 
        self.historial = []
        while(self.A.hash_pesos() != self.B.hash_pesos()):
            self.coordinaSalidas()
        print ('Sincronizadas en ' + str(time.time() - t_inicial)+ " segundos")   
        
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
        
    def graficoSincronizacion(self):
        plt.plot(range(len(self.historial)), self.historial)
          
        plt.xlabel('nº comunicacion')
        plt.ylabel('porcentaje sincronizacion')
        
        plt.show()
        
def prueba(n):         
    resS, resT, resC = [], [], []
    for i in range(n):       
        t = Test(5, 10, 10)
        (sincro, tiempo, cont) = t.fuerzaBruta(False)
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
     