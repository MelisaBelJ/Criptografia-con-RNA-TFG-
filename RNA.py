import funciones

class RNA:
    def __init__(self, funCoste = funciones.Errores.CuadraticoMedio(), funSalidaNorm = funciones.Funciones.Identidad().funcion):
        self.funCoste = funCoste
        self.funSalidaNorm = funSalidaNorm
        self.capas = []

    def anadeCapa(self, capa):
        self.capas.append(capa)

    def cambiaCoste(self, funCoste):
        self.funCoste = funCoste

    def propagacionHaciaDelante(self, entrada):
        salida = entrada
        for capa in self.capas:
            salida = capa.propagacionHaciaDelante(salida)
        self.salida = salida
        return salida
    
    def propagacionHaciaAtras(self, salidaEsperada, tasaAprendizaje = 0):
        error = self.funCoste.derivada(salidaEsperada, self.salida) #nabla A =(dC/da1, ... dC/daj)
        for capa in reversed(self.capas):
            error = capa.propagacionHaciaAtras(error, tasaAprendizaje)
        #return error
    
    def __call__(self, entrada):
        resultados = []
        for dato in entrada:
            resultados.append(self.propagacionHaciaDelante(dato))
        return list(map(self.funSalidaNorm, resultados))

    def entrena(self, entrada, salidaEsperada, iteraciones, tasaAprendizaje, imprime = False, imprimeCada = 100):
        numEntrada = len(entrada)
        
        for i in range(iteraciones):
            error = 0
            for j in range(numEntrada):
                salida = self.propagacionHaciaDelante(entrada[j]) #lo que obtenemos con los parametros actuales
                self.propagacionHaciaAtras(salidaEsperada[j], tasaAprendizaje) #Corrige parametros segun error        
                error += self.funCoste.funcion(salidaEsperada[j], salida) #acumula errores para luego hacer la media
            error /= numEntrada #media error
            
            if ((i % imprimeCada == (imprimeCada-1) or i==0) and imprime):            
                print('Iteracion n % d  error = % f' % (i+1, error))
        return error