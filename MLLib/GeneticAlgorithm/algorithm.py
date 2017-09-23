class GeneticAlgorithm:
    def __init__(popSize=10, maxGens=100000, crossoverRate=0.7, mutateRate=0.05):
        self.pop = []
        self.popSize = popSize
        self.maxGens = maxGens
        self.crossoverRate = crossoverRate
        self.mutateRate = mutateRate

    # Replace funcs
    def searialize(self):
        raise(NotImplementedError('Function searialize should be replaced'))

    def desearialize(self):
        raise(NotImplementedError('Function desearialize should be replaced'))

    def fitness(self):
        raise(NotImplementedError('Function fitness should be replaced'))

    def terminate(self):
        raise(NotImplementedError('Function terminate should be replaced'))

    # Specific funcs


    # General funcs
    def mutate(self, chromosome):
        return self.bitFlip(chromosome)

    def crossover(self, chromosome1, chromosome2):
        return self.onePointCrossover(chromosome1, chromosome2)

    def select(self, fitness):
        return self.selectRoulette(fitness)

    def _fitness(self, pop):
        for i in self.pop:
            pop[i] = (pop[i], self.fitness(i))
        return sorted(pop, key=itemgetter(1))

    def _mate(self):
        fittest = self._fitness(self.pop)
        a, b = self.select(fittest)
        return self.desearialize(self.mutate(self.crossover(self.searializeCreatue(a), self.searialize(b))))

    def reproduce(self):
        return self.reproduce

    # Interface funcs

    def initPop(seed):
        for i in seed:
            if len(self.pop) <= popSize:
                self.pop.append(i)
        while len(self.pop) <= popSize:
            self.pop.append(self.initCreature)

    def evolve():
        if len(self.pop) != self.popSize:
            self.initPop()
        if
