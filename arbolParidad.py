from RNA import RNA
import numpy
from capa import CapaUnoUno, CapaMult, CapaUnoUnoAtGeom
from funciones import reglaAprendizaje, Funciones, Errores
import hashlib

class arbolParidad(RNA):
    def __init__(self, k, n , l, caos = False, reglaA = reglaAprendizaje.AntiHebbian()):
        self.inicializa(k, n , l, reglaA)
        self.anadeCapa(CapaUnoUno(k, n, l, caos))
        self.anadeCapa(CapaMult(reglaA))
        self.caos = caos
        
    def inicializa(self, k, n , l, reglaA):
        self.l, self.k, self.n = l, k, n
        self.funCoste      = Errores.Multiplicacion()
        self.funSalidaNorm = Funciones.Identidad().funcion
        self.capas = []
        self.tasaAprendizaje, self.salida = 1, 0
        
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
    
    def getDatosIntermedios(self):
        return self.capas[0].salida   
    
    def cambiaDatosIntermedios(self, nuevosDatos):
        self.capas[0].salida = nuevosDatos
    
class arbolParidadAtGeom(arbolParidad):
    def __init__(self, k, n , l, caos = False, reglaA = reglaAprendizaje.AntiHebbian()):
        self.inicializa(k, n , l, reglaA)
        self.anadeCapa(CapaUnoUnoAtGeom(k, n, l, caos))
        self.anadeCapa(CapaMult(reglaA))
