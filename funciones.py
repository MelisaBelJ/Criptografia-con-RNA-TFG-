import numpy
from enum import Enum
from abc import ABCMeta, abstractmethod

class AbstractFuncion(metaclass = ABCMeta):     
    @abstractmethod
    def funcion(self, x):
        pass     
    @abstractmethod
    def derivada(self, x):
        pass 
    
class Identidad(AbstractFuncion):
    def funcion(self, x):
        return x    
    def derivada(self, x):
        return 0
    
class Sigmoide(AbstractFuncion):
    def funcion(self, x):
        return 1/(1 + numpy.exp(-x))    
    def derivada(self, x):
        return self.funcion(x)*(1.0-self.funcion(x))
    
class TangenteH(AbstractFuncion):
    def funcion(self, x):
        return numpy.tanh(x)
    def derivada(self, x):
        return 1-self.funcion(x)**2
    
class ReLu(AbstractFuncion):
    a = 1
    def funcion(self, x):
        return max(0,self.a*x)
    def derivada(self, x):
        return self.a
    def setA(self, a):
        self.a = a
  
class Swish(AbstractFuncion):
    a = 1
    s = Sigmoide()
    def funcion(self, x):
        return x*self.s.funcion(self.a*x)
    def derivada(self, x):
        y = self.funcion(x)
        return y+self.s(x)*(1- y)
    def setA(self, a):
        self.a = a
    
class PasoBinario(AbstractFuncion):
    paso = 0.5
    def funcion(self, x):
        return 0 if abs(x)<self.paso else 1
    def derivada(self, x):
        return 0
    def setPaso(self, paso):
        self.paso = paso

class Redondeo(AbstractFuncion):    
    digitos = 1    
    def funcion(self, x):
        return numpy.around(x, self.digitos)
    def derivada(self, x):
        return 0
    def setDigitos(self, digitos):
        self.digitos = digitos

class Logistica(AbstractFuncion):    
    r = 3.95
    def funcion(self, x):
        self.x = self.r*x*(1-x)
        return self.x
    def derivada(self, x):
        return self.r*(1-2*x)
    def setR(self, r):
        self.r = r
    def siguiente(self):
        return self.funcion(self.x)


class Paridad(AbstractFuncion):  
    def funcion(self, x):
        cont = 0
        for i in x:
            if i == 1:
                cont+=1
        if cont%2 == 0:
            return -1
        else:
            return 1
    def derivada(self, x):
        return 0

#Funciones de activacion
class Funciones(Enum):
    Identidad   = Identidad
    Sigmoide    = Sigmoide
    TangenteH   = TangenteH
    ReLu        = ReLu
    Swish       = Swish
    PasoBinario = PasoBinario
    Redondeo    = Redondeo
    Logistica   = Logistica
    Paridad     = Paridad
    def __call__(self):
        return self.value()
    
class AbstractFuncionError(metaclass = ABCMeta):     
    @abstractmethod
    def funcion(self, x, y):
        pass 
    @abstractmethod
    def derivada(self, x, y):
        pass 
    
class errorCuadraticoMedio(AbstractFuncionError):
    def funcion(self, actual, correcto):
        return numpy.mean(numpy.power(actual-correcto, 2))
    def derivada(self, actual, correcto):
        try:
            return 2*(correcto-actual)/actual.size
        except:
            return 2*(correcto-actual)       
        
class Multiplicacion(AbstractFuncionError):
    def funcion(self, actual, correcto):
        return actual*correcto
    def derivada(self, actual, correcto):
        return actual*correcto
    
#Funciones de error
class Errores(Enum):
    CuadraticoMedio = errorCuadraticoMedio
    Multiplicacion  = Multiplicacion
    def __call__(self):
        return self.value() 
   
#Reglas para la estructura TPM
class reglaAprendizaje(Enum):
    Hebbian     = "Hebbian"
    AntiHebbian = "Anti Hebbian"
    RandomWalk  = "Random Walk"
    def __call__(self):
        return self.name