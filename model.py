# A toy RNN-LSTM implementation by Rex Wang
# Following the idea of http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

import copy
try:
    import numpy as np
except:
    print("There is something wrong with your numpy!")


class RNN_model(object):

    def __init__(self, learning_rate=0.1,
                 input_dim=2,
                 hidden_dim=16,
                 output_dim=1,
                 iteration=100000, binary_dim=8):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.iteration = iteration
        self.binary_dim = binary_dim
        self.int2binary = {}
        # Generating training dataset

        # Computing the binary presentation of 1-256
        self.largest_number = np.power(2, self.binary_dim)
        binary = np.unpackbits(np.array([range(self.largest_number)], dtype=np.uint8).T, axis=1)
        for i in range(self.largest_number):
            self.int2binary[i] = binary[i]

        # Neural network weights initialization
        self.w_0 = 2*np.random.random((self.input_dim, self.hidden_dim)) - 1
        self.w_1 = 2*np.random.random((self.hidden_dim, self.output_dim)) - 1
        self.w_h = 2*np.random.random((self.hidden_dim, self.hidden_dim)) - 1

        self.w_0_update = np.zeros_like(self.w_0)
        self.w_1_update = np.zeros_like(self.w_1)
        self.w_h_update = np.zeros_like(self.w_h)

    def sigmoid(self, x):
        # To ensure the stability of numeraical computation
        # We are not using this: return 1/(1+np.exp(-x))
        return np.exp(-np.logaddexp(0, -x))

    def sigmoid_derive(self, x):
        return np.multiply(x, (1-x))

    def run(self):
        for j in range(self.iteration):

            # Generate a simpple add problem (a+b=c)
            a_int = np.random.randint(self.largest_number/2)
            a = self.int2binary[a_int]

            b_int = np.random.randint(self.largest_number/2)
            b = self.int2binary[b_int]

            # Correct answer
            c_int = a_int + b_int
            c = self.int2binary[c_int]

            # Predicted value of network
            d = np.zeros_like(c)

            # Reset error to 0 for each iteration
            overallError = 0

            layer_2_deltas = list()
            layer_1_values = list()
            layer_1_values.append(np.zeros(self.hidden_dim))

            # ----- Feedforward pass -----

            # Moving along the positions in the binary encoding
            for position in range(self.binary_dim):

                # Generate input and output
                X = np.array(
                    [[a[self.binary_dim - position - 1], b[self.binary_dim - position - 1]]])
                y = np.array([[c[self.binary_dim - position - 1]]]).T

                # Hidden layer
                # Input_layer + pervious_hidden_layer = new_hidden_layer
                layer_1 = self.sigmoid(np.dot(X, self.w_0) + np.dot(layer_1_values[-1], self.w_h))

                # Output layer
                layer_2 = self.sigmoid(np.dot(layer_1, self.w_1))

                # Error
                layer_2_error = y - layer_2
                layer_2_deltas.append((layer_2_error) * self.sigmoid_derive(layer_2))
                overallError += np.abs(layer_2_error[0])

                # Decode estimate
                d[self.binary_dim - position - 1] = np.round(layer_2[0][0])

                # Store hidden layer
                layer_1_values.append(copy.deepcopy(layer_1))

            future_layer_1_delta = np.zeros(self.hidden_dim)

            # ----- Backpropagation pass -----
            for position in range(self.binary_dim):

                X = np.array([[a[position], b[position]]])
                layer_1 = layer_1_values[-position - 1]
                prev_layer_1 = layer_1_values[-position - 2]

                # Error at output layer
                layer_2_delta = layer_2_deltas[-position - 1]

                # Error at hidden layer
                layer_1_delta = (future_layer_1_delta.dot(
                    self.w_h.T) + layer_2_delta.dot(self.w_1.T)) * self.sigmoid_derive(layer_1)

                self.w_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                self.w_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                self.w_0_update += X.T.dot(layer_1_delta)

                future_layer_1_delta = layer_1_delta

            # Update transform matrices
            self.w_0 += self.w_0_update * self.learning_rate
            self.w_1 += self.w_1_update * self.learning_rate
            self.w_h += self.w_h_update * self.learning_rate

            # Reset all matrices after each iteration
            self.w_0_update *= 0
            self.w_1_update *= 0
            self.w_h_update *= 0

            # Printing the progress
            if(j % 1000 == 0):
                print('========================')
                print("Training with")
                print("learning rate: %s" % str(self.learning_rate))
                print("hidden size: %s" % str(self.hidden_dim))
                print("current iteration: %s" % str(j))

                print('------------------------')
                print("Overall Error: %s" % str(overallError))
                print("Pred: %s" % str(d))
                print("True: %s" % str(c))
                out = 0
                for idx, x in enumerate(reversed(d)):
                    out += np.multiply(x, np.power(2, idx))
                print("%s + %s = %s" % (str(a_int), str(b_int), str(out)))
                print('')

if __name__ == '__main__':
    rnn = RNN_model()
    rnn.run()
