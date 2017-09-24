import neuralnet as n
import numpy as np

net = n.NeuralNet([2, 4, 1])

net.train(np.array([[0, 1]]), np.array([[1]]))
