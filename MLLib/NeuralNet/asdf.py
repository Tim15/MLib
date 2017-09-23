class NeuralNet:
  def init(self, layers):
      self.network = [2*np.random.random((layers[i]+1,layers[i+1]))-1 for i in range(len(layers)-1)]
    # set netowrk up

  def activate(self, input, d = False):
    # this can be overridden

  def train(self, input, expected, iterations=100000, log):
      for i in range(iterations):
          outputs = [input]
          for layer in network:
              outputs[-1] = np.c_[outputs[-1], np.ones(len(outputs[-1]))]
              outputs.append(nonlin(np.dot(outputs[-1], layer)))
    #  forward and back propagate and log

  def predict(self, input):
    # forward propagate

  def evolve(self):
    # evolve

  def serealize(self):
    # pass

  def desearealize(self, input):
    # pass
