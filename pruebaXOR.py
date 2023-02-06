import numpy
from capa import CapaConectada
from funciones import Funciones
from RNA import RNA

entrenamientoEntrada = numpy.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
entrenamientoSalida = numpy.array([[[0]], [[1]], [[1]], [[0]]])

numpy.random.seed(2)

red = RNA(funSalidaNorm = Funciones.PasoBinario().funcion)
red.anadeCapa(CapaConectada(2, 3, Funciones.TangenteH()))
red.anadeCapa(CapaConectada(3, 1, Funciones.TangenteH()))

red.entrena(entrenamientoEntrada, entrenamientoSalida, 500, 0.1, True, imprimeCada = 100)

salida = red(entrenamientoEntrada)
print(salida)