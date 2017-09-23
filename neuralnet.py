class NeuralNet:
    def init(self, layers): # layers = [0,1]
        self.network = [2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1 for i in range(len(layers) - 1)]
      # set netowrk up

    def activate(self, input, d=False):
      # this can be overridden

    def train(self, input, expected, iterations=100000, log):
        for i in range(iterations):
            outputs = forwardPropagate(input)
      #  forward and back propagate and log

    def forwardPropagate(input):
        outputs = [input]
        for layer in self.network:
            outputs[-1] = np.c_[outputs[-1], np.ones(len(outputs[-1]))]
            outputs.append(self.activate(np.dot(outputs[-1], layer)))
        return outputs

    def backPropagate(input):
        outputs = [input]
        for layer in self.network:
            outputs[-1] = np.c_[outputs[-1], np.ones(len(outputs[-1]))]
            outputs.append(nonlin(np.dot(outputs[-1], layer)))
        return outputs

    def predict(self, input):
      # forward propagate
      return forwardPropagate(input)

    def evolve(self):
      # evolve

    def serealize(self):
      # pass

    def desearealize(self, input):
      # pass
