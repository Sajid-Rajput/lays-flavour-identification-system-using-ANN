import numpy as np

X = np.array(([[0.20255932, 0.73943446, 0.90064627, 0.30740877, 0.09450361],
 [0.67189936, 0.26836411, 0.39191808, 0.14912873, 0.2482556 ],
 [0.55067187, 0.4254567 , 0.44619548, 0.21762929, 0.20541069],
 [0.53007175, 0.41629071, 0.47984474, 0.23670435, 0.75970132],
 [1.        , 0.        , 0.        , 0.        , 0.        ],
 [0.09412381, 0.84002353, 0.78733817, 0.58949611, 0.08752601],
 [0.        , 0.68036551, 0.84410164, 1.        , 1.        ],
 [0.50700083, 0.35749056, 0.51140844, 0.378915  , 0.07124495],
 [0.07561085, 1.        , 1.        , 0.19323977, 0.18606928],
 [0.73856247, 0.15168126, 0.5049619 , 0.0179836 , 0.0314604 ]]))

Y = np.array(([[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 1, 0, 0],
]))


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Sigmoid_Activation:
    def forward(self, inputs):
        self.output =  1/(1 + np.exp(-(inputs)))


layer1 = Layer_Dense(5,5)
layer2 = Layer_Dense(5, 5)
layer3 = Layer_Dense(5,5)
activation1 = Sigmoid_Activation()
activation2 = Sigmoid_Activation()
activation3 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation2.forward(layer3.output)


lowest_accuracy = 0 
best_layer1_weights = layer1.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases = layer2.biases.copy()

for iteration in range(100000):
    layer1.weights += 0.05 * np.random.randn(5, 5)
    layer1.biases += 0.05 * np.random.randn(1, 5)
    layer2.weights += 0.05 * np.random.randn(5, 5)
    layer2.biases += 0.05 * np.random.randn(1, 5)
    layer3.weights += 0.05 * np.random.randn(5, 5)
    layer3.biases += 0.05 * np.random.randn(1, 5)

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    predictions = np.argmax(activation2.output, axis=1)
    target = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == target) * 100

    if accuracy > lowest_accuracy:
        print("New set of weights founds, iteration:", iteration, "accuracy: ", accuracy,"%")
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()
        lowest_accuracy = accuracy
    else:
        layer1.weights = best_layer1_weights
        layer1.biases = best_layer1_biases
        layer2.weights = best_layer2_weights
        layer2.biases = best_layer2_biases


print("Best Layer1 weights: ", best_layer1_weights)
print("Best Layer1 biases: ", best_layer1_biases)
print("Best Layer1 weights: ", best_layer2_weights)
print("Best Layer2 biases: ", best_layer2_biases)
print("Lowest loss: ", lowest_loss)
