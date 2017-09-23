import neural
import numpy as np

# class Xor:
    # def __init__(self):
    #     np.random.seed(1)
    #     self.network = [np.random.random((3, 4)), np.random.random((5, 1))] # [(2 input nodes + 1 bias node) X (4 hidden nodes), (4 hidden nodes + 1 bias) X (1 output node)]
    #     # print(self.network[1])
    # def train(self, inputData):
    #     def sig(x, d=False):
    #         if d:
    #             return (x*(1-x))
    #         return 1/(1+np.exp(-x))
    #     inputData, prediction = np.array(inputData[0]), np.array(inputData[1])
    #     for iteration in range(100000):
    #         # outputs = [[a, b] for a, b in inputData]
    #         outputs = [inputData]
    #         outputs.append(sig(np.dot(np.c_[outputs[-1], np.ones(len(outputs[-1]))], self.network[0])))
    #         outputs.append(sig(np.dot(np.c_[outputs[-1], np.ones(len(outputs[-1]))], self.network[1])))
    #         # print('outputs, prediction', outputs, prediction, outputs[-1].shape)
    #         print('weights', self.network)
    #         errors = [prediction - outputs[-1]]
    #         deltas = [errors[-1] * sig(outputs[2])]
    #         # if(iteration % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
    #         #     print("Error: " + str(np.mean(np.abs(errors[-1]))))
    #         # print('Layer 2 error', errors[-1])
    #         print('Layer 2 delta', deltas[-1])
    #         # print('Layer 2 delta, weights', deltas[-1], self.network[1])
    #         errors.insert(0, deltas[-1].dot(self.network[1].T))
    #         deltas.insert(0, errors[-1] * sig(outputs[1],d=True))
    #         # print('Layer 1 error', errors[0])
    #         print('Layer 1 delta', deltas[0])
    #
    #
    #         for i in range(len(deltas)):
    #             self.network[1] += outputs[1].T.dot(deltas[1])
    #             self.network[0] += outputs[0].T.dot(deltas[0])
    # def output(self, inputData):
    #     output = [inputData + [1]]
    #     output = sig(np.dot(self.network[0], output))
    #     return(sig(np.dot(self.network[1], output)))

testData = ([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
neuralNet = neural.NeuralNet([2, 4, 1])
# print(neuralNet.forwardPropagate([[0, 1]]))
# neuralNet = Xor()
neuralNet.train(testData[0], testData[1])
# print([neuralNet.output((i, v)) for i in testData])
