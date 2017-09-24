import numpy as np

class NeuralNet:
    def __init__(self, layers): # layers = [0,1]
        self.network = [2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1 for i in range(len(layers) - 1)]
        print(self.network)
      # set netowrk up

    def activate(self, x, d=False):
      # this can be overridden
      if d:
          return x * (1-x)
      return 1/(1 + np.exp(-x))

    def train(self, input, expected, iterations=100000):
        for i in range(iterations):
            print("Iteration #" + str(i))
            outputs = self.forwardPropagate(input) # [np.ndarray ...], outputs of each layer
            print(outputs)
            error = self.calculateErrors(outputs, expected) # [np.ndarray ...] error for each layer, starting from the last
            print("Error: " + error[0])
            self.backPropagate(outputs, error)
      #  forward and back propagate and log

    def forwardPropagate(self, input):
        outputs = [input]
        for layer in self.network:
            outputs[0] = np.c_[outputs[0], np.ones(len(outputs[0]))]
            print(outputs)
            print(layer)
            outputs.insert(0, self.activate(np.dot(outputs[0], layer)))
        return outputs

    def calculateErrors(self, outputs, expected):
        err = []
        # final layer
        err.append(self.error(expected, outputs[-1]))
        # iterate from second-to-last back to first layer
        for i in reversed(range(len(self.network)-1)):
            layer = self.network[i]
            err.append(np.dot(layer.T, err[-1]))
        return err

    def error(self, expected, got):
        return expected - got

    def backPropagate(self, output, err):
        for i in reversed(range(self.network)):
            self.network[i] += self.learningRate * np.dot((err[i] * output[i] * (1.0 - output[i])), np.transpose(output[i-1]))

        return outputs

    def predict(self, input):
      # forward propagate
      return forwardPropagate(input)

    def evolve(self):
      # evolve
      pass

    def serealize(self):
      # pass
      pass

    def desearealize(self, input):
      # pass
      pass
