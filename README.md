RNN-ToyNet
=========================

This ToyNet is a really simple implementation of Recurrent Neural Network, it is so simple that the only extra dependency of it is numpy, no need to touch Tensorflow or Theano or PyTorch. But this is a really good example for anyone who want to get in touch with RNN. This implementation is following the idea of [iamtrask's blog](http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/), but with some modifications to make the computations stable and the code clean.

This RNN is used to deal with a "Binary Addition" problem as showed below:
<p align="center">
  <img src="https://github.com/rexwangcc/RNN-ToyNet/blob/master/binary_addition.GIF"/>
</p>



Below are explanations of the code, also referred from the original blog:

>Lines 0-8: Importing our dependencies and seeding the random number generator. We will only use numpy and copy. Numpy is for matrix algebra. Copy is to copy things. If there is something wrong with numpy, it will raise an error.
>Lines 11: Create a class of this RNN model, to make parameters flexible.
>Lines 13-24: Constructor of our class, stores our input parameters.

>Lines 28-40: Initializing our weights that will be used in later process.
>Lines 29: We're going to create a lookup table that maps from an integer to its binary representation. The binary representations will be our input and output data for each math problem we try to get the network to solve. This lookup table will be very helpful in converting from integers to bit strings.

>Lines 42-48: Our nonlinearity and derivative. To ensure the computational statbility(although it is not necessary here for this simple network), used another version from the original function version.

>Input_dim: We are adding two numbers together, so we'll be feeding in two-bit strings one character at the time each. Thus, we need to have two inputs to the network (one for each of the numbers being added).
>
>hidden_dim: This is the size of the hidden layer that will be storing our carry bit. Notice that it is way larger than it theoretically needs to be. Play with this and see how it affects the speed of convergence. Do larger hidden dimensions make things train faster or slower? More iterations or fewer?
>
>output_dim: Well, we're only predicting the sum, which is one number. Thus, we only need one output
>
>w_0: This is the matrix of weights that connects our input layer and our hidden layer. Thus, it has "input_dim" rows and "hidden_dim" columns. (2 x 16 unless you change it). If you forgot what it does, look for it in the pictures in Part 2 of this blogpost.
>
>w_1: This is the matrix of weights that connects the hidden layer to the output layer. Thus, it has "hidden_dim" rows and "output_dim" columns. (16 x 1 unless you change it). If you forgot what it does, look for it in the pictures in Part 2 of this blogpost.
>
>w_h: This is the matrix of weights that connects the hidden layer in the previous time-step to the hidden layer in the current timestep. It also connects the hidden layer in the current timestep to the hidden layer in the next timestep (we keep using it). Thus, it has the dimensionality of "hidden_dim" rows and "hidden_dim" columns. (16 x 16 unless you change it). If you forgot what it does, look for it in the pictures in Part 2 of this blogpost.
>
>Line 38-40: These store the weight updates that we would like to make for each of the weight matrices. After we've accumulated several weight updates, we'll actually update the matrices. More on this later.
>
>Line 50: We're iterating over 100,000 training examples (iterations) by default, but since now we define the training process in a method contained in the RNN model class, and the varaible "iteration" can be changed during instantiate.
>
>Line 54: We're going to generate a random addition problem. So, we're initializing an integer randomly between 0 and half of the largest value we can represent. If we allowed the network to represent more than this, than adding two number could theoretically overflow (be a bigger number than we have bits to represent). Thus, we only add numbers that are less than half of the largest number we can represent.
>
>Line 54: We lookup the binary form for "a_int" and store it in "a"
>
>Line 57: Same thing as line 54, just getting another random number.
>
>Line 58: Same thing as line 55, looking up the binary representation.
>
>Line 61: We're computing what the correct answer should be for this addition
>
>Line 62: Converting the true answer to its binary representation
>
>Line 65: Initializing an empty binary array where we'll store the neural network's predictions (so we can see it at the end). You could get around doing this if you want...but i thought it made things more intuitive
>
>Line 68: Resetting the error measure (which we use as a means to track convergence... see my tutorial on backpropagation and gradient descent to learn more about this)
>
>Lines 70-71: These two lists will keep track of the layer 2 derivatives and layer 1 values at each time step.
>
>Line 72: Time step zero has no previous hidden layer, so we initialize one that's off.
>
>Line 77: This for loop iterates through the binary representation
>
>Line 80: X is the same as "layer_0" in the pictures. X is a list of 2 numbers, one from a and one from b. It's indexed according to the "position" variable, but we index it in such a way that it goes from right to left. So, when position == 0, this is the farhest bit to the right in "a" and the farthest bit to the right in "b". When position equals 1, this shifts to the left one bit.
>
>Line 82: Same indexing as line 80, but instead it's the value of the correct answer (either a 1 or a 0)
>
>Line 86: This is the magic!!! Make sure you understand this line!!! To construct the hidden layer, we first do two things. First, we propagate from the input to the hidden layer (np.dot(X,synapse_0)). Then, we propagate from the previous hidden layer to the current hidden layer (np.dot(prev_layer_1, synapse_h)). Then WE SUM THESE TWO VECTORS!!!!... and pass through the sigmoid function.
>
>So, how do we combine the information from the previous hidden layer and the input? After each has been propagated through its various matrices (read: interpretations), we sum the information.
>
>Line 89: This should look very familiar. It's the same as previous tutorials. It propagates the hidden layer to the output to make a prediction
>
>Line 92: Compute by how much the prediction missed
>
>Line 93: We're going to store the derivative (mustard orange in the graphic above) in a list, holding the derivative at each timestep.
>
>Line 94: Calculate the sum of the absolute errors so that we have a scalar error (to track propagation). We'll end up with a sum of the error at each binary position.
>
>Line 97 Rounds the output (to a binary value, since it is between 0 and 1) and stores it in the designated slot of d.
>
>Line 100 Copies the layer_1 value into an array so that at the next time step we can apply the hidden layer at the current one.
>
>Line 105: So, we've done all the forward propagating for all the time steps, and we've computed the derivatives at the output layers and stored them in a list. Now we need to backpropagate, starting with the last timestep, backpropagating to the first
>
>Line 107: Indexing the input data like we did before
>
>Line 108: Selecting the current hidden layer from the list.
>
>Line 109: Selecting the previous hidden layer from the list
>
>Line 112: Selecting the current output error from the list
>
>Line 115: this computes the current hidden layer error given the error at the hidden layer from the future and the error at the current output layer.
>
>Line 118-120: Now that we have the derivatives backpropagated at this current time step, we can construct our weight updates (but not actually update the weights just yet). We don't actually update our weight matrices until after we've fully backpropagated everything. Why? Well, we use the weight matrices for the backpropagation. Thus, we don't want to go changing them yet until the actual backprop is done. See the backprop blog post for more details.
>
>Line 125 - 132 Now that we've backpropped everything and created our weight updates. It's time to update our weights (and empty the update variables).
>
>Line 135 - end Just some nice logging to show progress>