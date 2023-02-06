from RNA import RNA
import numpy
from capa import CapaUnoUno, CapaMult
from funciones import reglaAprendizaje, Funciones, Errores
import hashlib

class arbolParidad(RNA):
    def __init__(self, k, n , l, reglaA = reglaAprendizaje.AntiHebbian()):
        self.funCoste = Errores.Multiplicacion()
        self.funSalidaNorm = Funciones.Identidad().funcion
        self.capas = []
        self.k = k
        self.n = n
        self.l = l
        self.tasaAprendizaje = 0
        self.anadeCapa(CapaUnoUno(k, n, l))
        self.anadeCapa(CapaMult(reglaA))
        self.salida = 0
        
    def cambiaReglaAprendizaje(self, reglaA):
        self.capas[1].cambiaReglaAprendizaje(reglaA)
        
    def getPesos(self):
        return self.capas[0].pesos.copy()
    
    def __call__(self, entrada):
            return self.propagacionHaciaDelante(entrada)
    
    def sincronizacionCon(self, otra):
    	return 100 - numpy.average(100 * numpy.abs(self.getPesos() - otra.getPesos())/(2*self.l - 1))
    
    def hash_pesos(self):
        return hashlib.sha256((f"{self.salida}{self.getPesos()}").encode('UTF-8')).hexdigest()