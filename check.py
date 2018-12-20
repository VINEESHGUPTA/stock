import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



data = pd.read_csv("AXISBANK.csv")    # data from 30 oct to 30 nov 2017 of Axis bank
X = data[['Open', 'High', 'Low']]
y = data[['Close']]
xPredicted = np.array(([539.40, 539.90, 531.90]), dtype=float)    #giving input of 1 dec 2017

X = X/np.amax(X, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)
y = y/1000

class Neural_Network(object):
  def __init__(self):
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 3
    self.learningRate = 0.1


    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

  def forward(self, X):

    self.z = np.dot(X, self.W1)
    self.z2 = self.sigmoid(self.z)
    self.z3 = np.dot(self.z2, self.W2)
    o = self.sigmoid(self.z3)
    return o

  def sigmoid(self, s):

    return 1 / (1 + np.exp(-s))


  def sigmoidPrime(self, s):

    return s * (1 - s)

  def backward(self, X, y, o):

    self.o_error = -(y - o)
    self.o_delta = self.o_error*self.sigmoidPrime(o)

    self.z2_error = self.o_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

    self.W1 -= (self.learningRate * X.T.dot(self.z2_delta))
    self.W2 -= (self.learningRate * self.z2.T.dot(self.o_delta))

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(xPredicted))
    print("Output: \n" + (str(self.forward(xPredicted)*1000)))
    print("Loss: \n" + str(np.mean(np.square(y - NN.forward(xPredicted)))))


NN = Neural_Network()
u = []
v = []
for i in range(100):# train 1000 times
  u.append(i)
  v.append(np.mean(np.square(y - NN.forward(xPredicted))))
  NN.train(X, y)

# NN.saveWeights()
NN.predict()
plt.plot(u,v)
plt.show()
