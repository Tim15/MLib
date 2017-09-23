import numpy as np
import activation as act
import cost

# class Network:
#     def __init__(self, inputs, hiddenLayers, outputs, learningRate=0.01):
#         np.random.seed(1)
#         self.learningRate = learningRate
#         # layers = [[inputs]] + hiddenLayers + [[outputs]]
#         # self.network = [[0 for i in range(inputs)]]
#         # for layer in range(len(hiddenLayers)):# iterate over the layers
#         #     self.network[layer].append([])
#         #     for node in range(len(layer)): # to node
#         #         self.network[layer].append(Node(self.sig, self.sig_derivative, [np.random.random() for i in layers[layer]], np.random.random(), self.learningRate))
#         layers = hiddenLayers + [outputs]
#         self.network = [2*np.random.random((layers[index-1]+1 if index > 0 else inputs+1, layers[index]))-1 for index in range(len(layers))]
#         print(self.network)
#
#     def sig(self, x):
#         return 1/(1+np.exp(-x))
#     def sig_derivative(self, x):
#         return x*(1-x)
#     # def tanh(x, derivitive=False):
#     #     pass
#
#     def feedForward(self, outputs):
#         outputs = [list(outputs)+[1]]
#         for layer in self.network:
#             # print(outputs, np.r_[outputs[-1],[1]])
#             outputs.append(np.c_[self.sig(np.dot(outputs[-1], layer)),[1]])
#         return outputs
#
#     def getError(self, output, prediction):
#         if type(output) == list and type(prediction) == list:
#             return sum([(i-v)**2 for i,v in zip(output, prediction)])/2
#         return sum(output[-1])-prediction
#
#     # def learnSynapse(self)
#
#     def backProp(self, inputs, answer, log=False):
#         outputs = self.feedForward(inputs)[1:]
#         error = [self.getError(outputs, answer)]
#         deltas = [error*self.sig_derivative(outputs[-1])]
#         if log:
#             print('error', error[0])
#
#         for layer in reversed(range(len(self.network)-1)):
#             error.insert(0, deltas[0].dot(self.sig_derivative(self.network[layer+1].T)))
#             # print(deltas[0], self.network[layer+1].T, error[0], outputs[layer])
#             deltas.insert(0, error[0]*self.sig_derivative(np.r_[outputs[layer], [1]]))
#
#         for i in range(len(deltas)):
#             # print(len(deltas), len(self.network), len(outputs))
#             print(outputs[i], deltas[i], self.network[i])
#             self.network[i] += outputs[i].T.dot(deltas[i])
#             # self.network[i] += outputs[i].T.dot(deltas[i])
#
#
#     def learnCase(self, inputs, answer, iterations=10000):
#         for i in range(iterations):
#             self.backProp(inputs, answer, log=(i%(iterations//10)==0))
#
#     def _parse(self, data):
#         return zip(data[0], data[1])
#
#     def train(self, data):
#         for input, answer in self._parse(data):
#             self.learnCase(input, answer)
#
#     def output(self, inputs):
#         return self.feedForward(inputs)[-1]
#
#     def exportNet(self):
#         pass
#
#     def importNet(self):
#         pass

class NeuralNet:
    def __init__(self, layers, activators=None, l=0.9, m=0.01):
        np.random.seed(1)
        self.learningRate = l
        self.network = [np.random.random((layers[i]+1, layers[i+1])) for i in range(len(layers)-1)]
        print('Weights:', self.network, '\n')

    def cost(self, output, predicted):
        return cost.sumSq(output, expected)

    def forwardPropagate(self, inputs):
        outputs = [inputs]
        for weights in self.network:
            outputs[-1] = np.c_[outputs[-1], np.ones(len(outputs[-1]))]
            outputs.append(act.sigmoid(np.dot(outputs[-1], weights)))
        print('outputs', outputs, '\n')
        return outputs

    def backPropagate(self, inputs, predicted, log=True):
        outputs = self.forwardPropagate(inputs)
        errors = [predicted-outputs[-1]]
        print(errors[0])
        deltas = [act.sigmoid(outputs[-1], d=True) * errors[-1] * self.learningRate]
        print(deltas[0])
        for i in range(len(self.network)):
            print()
            errors.insert(0, deltas[0].dot(self.network[1].T))
            print(errors[0])
            deltas.insert(0, errors[0]*act.sigmoid(outputs[i+1], d=True))
            print(deltas[0])

        for i in range(len(deltas)):
            self.network[i] += deltas[i] * outputs[i]

    def train(self, inputs, predicted, passes=10000, log=True):
        for i in range(passes):
            logPass = passes%(i+1) == 0
            self.backPropagate(inputs, predicted, log=logPass)
        return self.output(inputs)

    def output(self, input):
        return self.forwardPropagate(inputs)[-1]


"""
I1 I2

H1 H2 H3 H4

O1

Input = [I1, I2, 1]
Weights = [
    [[w1, w2, w3, w4],
    [w5, w6, w7, w8],
    [b1, b2, b3, b4]],

    [[w9],
    [w10],
    [w11],
    [w12],
    [b6]]]
Output = [[I1, I2, 1],
    Output[0] * weights[0],
    output[1] * weights[1]
Error = []
Delta = []
L = learning rate
A(X) = 1/(1-e^-x)
error(output, predicted) = 1/2 * sum([(prediected[i]-output[i] )^2 for i in output])

Error[1] = Error(Output[2], predicted)
Delta[1] = Error[1] * A'(Output[2])

for index in Weights[:-1].Reversed:
    Error[index] = Delta[index+1].dot(A'(Weights[index+1].Transpose))
    Delta[index] = Error[index] * A'(Output[index])

for index in Delta:
    Weights[index] += Delta[index]
"""
