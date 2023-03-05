import numpy
from funciones import Funciones
from abc import ABCMeta, abstractmethod

class AbstractCapa(metaclass = ABCMeta):
    def __init__(self):
        self.entrada = None
        self.salidaPreActivacion = None

    @abstractmethod
    def propagacionHaciaDelante(self, entrada):
        raise NotImplementedError

    @abstractmethod
    def propagacionHaciaAtras(self, errorSalida, tasaAprendizaje):
        raise NotImplementedError

#Capa para una red con propagacion hacia delante
class CapaConectada(AbstractCapa):
    def __init__(self, numNeuronasEntrada, numNeuronasSalida, funcionActivacion = Funciones.Identidad()):
        self.funcionActivacion = funcionActivacion
        self.pesos = numpy.random.rand(numNeuronasEntrada, numNeuronasSalida) - 0.5
        self.sesgos = numpy.random.rand(1, numNeuronasSalida) - 0.5

    #devuelve sigma(xW+b) con x = entrada, W = pesos, b = sesgos
    def propagacionHaciaDelante(self, entrada):
        self.entrada = entrada
        self.salidaPreActivacion = numpy.dot(self.entrada, self.pesos) + self.sesgos #xW+b
        return self.funcionActivacion.funcion(self.salidaPreActivacion) #sigma(xW+b)

    # error salida: nabla A =(dC/da1, ... dC/daj)
    def propagacionHaciaAtras(self, errorSalida, tasaAprendizaje):
        delta = self.funcionActivacion.derivada(self.salidaPreActivacion) * errorSalida # k = nabla A sigma'(Z)
        errorEntrada =  numpy.dot(delta, self.pesos.T)# kW
        self.pesos -= tasaAprendizaje * numpy.dot(self.entrada.T, delta) # la(xk)
        self.sesgos -= tasaAprendizaje * delta # mk
        
        return errorEntrada        

#Capa en la que se conecta cada neurona con una unica neurona de la siguiente capa, cada una diferente. Usa como funcion activacion la de paridad (para TPM)
class CapaUnoUno(AbstractCapa):
    def __init__(self, numNeuronasEntrada, n, l, caos = False):
        self.l, self.caos = l, caos
        self.pesos = numpy.random.randint(-l, l + 1, [numNeuronasEntrada, n])

    #devuelve sigma(xW) con x = entrada, W = pesos
    def propagacionHaciaDelante(self, entrada):
        self.entrada = entrada
        self.salidaPreActivacion = numpy.sum(self.entrada * self.pesos, axis = 1) #xW
        if self.caos:
            self.salidaPreActivacion = Funciones.Logistica(self.salidaPreActivacion)
        self.salida = [1 if i == 0 else i for i in numpy.sign(self.salidaPreActivacion)] #sigma(xW)
        return self.salida

    def propagacionHaciaAtras(self, errorSalida, tasaAprendizaje):
        for i in range(len(self.pesos)):
            for j in range(len(self.pesos[0])):
                self.pesos[i,j] = numpy.clip(self.pesos[i, j] + (errorSalida[i]*self.entrada[i,j]), -self.l, self.l)
        return 0  

#Multiplica todas las salidas entre si y devuelve el resultado (para TPM)    
class CapaMult(AbstractCapa):
    def __init__(self, reglaA):
        self.reglaAprendizaje = reglaA
    
    def cambiaReglaAprendizaje(self, reglaA):
        self.reglaAprendizaje = reglaA

    def propagacionHaciaDelante(self, entrada):
        self.entrada = entrada
        self.salida  = numpy.prod(entrada)
        return self.salida

    def propagacionHaciaAtras(self, errorSalida, tasaAprendizaje):
        if errorSalida <= 0:
            return [0 for _ in self.entrada]
        else:
            rA = self.reglaAprendizaje
            if rA == "Random Walk":
                return [1 if self.salida == i else 0 for i in self.entrada]
            else:
                aux = -1 if rA == "Hebbian" else 1
                return [(aux*self.salida) if self.salida*i>0 else 0 for i in self.entrada]
         

#Capa en la que se conecta cada neurona con una unica neurona de la siguiente capa, cada una diferente. Usa como funcion activacion la de paridad (para TPM)
class CapaUnoUnoAtGeom(CapaUnoUno):
    def propagacionHaciaAtras(self, errorSalida, tasaAprendizaje):
        noCoincide, i = True, 0
        while noCoincide and i< len(errorSalida):
             noCoincide = errorSalida[i] == 0
             i += 1
        if noCoincide:
            iMin, salMin = 0, self.salidaPreActivacion[0]
            for i in range(1, len(self.pesos)):
                if salMin > self.salidaPreActivacion[i]:
                    iMin, salMin = i, self.salidaPreActivacion[i]
            for j in range(len(self.pesos[0])):
                self.pesos[iMin,j] = numpy.clip(self.pesos[iMin, j] - (errorSalida[iMin]*self.entrada[iMin,j]), -self.l, self.l)
        else:
              for i in range(len(self.pesos)):
                  for j in range(len(self.pesos[0])):
                      self.pesos[i,j] = numpy.clip(self.pesos[i, j] + (errorSalida[i]*self.entrada[i,j]), -self.l, self.l)
              
        return 0  
              
