# Baby TensorFlow (1.0)

A homemade machine learning neural network architecture modeled after TensorFlow. 

### Installation

To install Baby_TensorFlow, use the command:

    pip install git+https://github.com/WilliamBidle/Baby-TensorFlow

To test the installation, run the following code into your Python editor of choice:

    from Baby_TensorFlow import * 
    
    layer_sequence = [1,'ReLU', 2, 'sigmoid', 3]
    loss_function = 'MSLE'

    nn = NN(layer_sequence, loss_function)

    print('activation func library:\n', nn.activation_funcs_library, '\n')
    print('loss func library:\n', nn.loss_funcs_library, '\n')
    print('current weights:\n', nn.weights, '\n')
    print('current activation functions:\n', nn.activation_funcs, '\n')
    print('current loss function:\n', nn.loss_func_label, ':', nn.loss_func, '\n')
    print('traing error:\n', nn.training_err, '\n')

If there are no errors, then you have successfully installed Baby_TensorFlow! The full list of functions, their usage, as well as some examples can be found within the ***Baby_Tensorflow.py*** file.

### List of available activation functions:

- "sigmoid" : 

$$\frac{1}{1 + e^{-x}}$$

- 'tanh' : 

$$tanh(x)$$

- 'ReLU' : 

$$f(x) = \begin{cases}
x & \text{if } x \geq 0,\\
0  & \text{if } x < 0.
\end{cases}$$

### List of avaliable loss functions:

For a given network output vector, $\vec{y}^{out}$, and true value vector, $\vec{y}^{true}$, with $N$ components each, different loss functions are definined by the following.

- Mean Squared Error ("MSE") : 

$$\sum_{i}^N(y_i^{out} - y_i^{true})^2$$

- Mean Absolute Error ("MAE") : 

$$\sum_{i}^N|y_i^{out} - y_i^{true}|$$

- "MAPE" : 
$$100 * \sum_{i}^N|\frac{y_i^{out} - y_i^{true}}{y_i^{out} + y_i^{true}}|$$

- Mean Squared Logarithmic Error ("MSLE") : 

$$\sum_{i}^N(log(y_i^{out} + 1) - log(y_i^{true} + 1))^2$$ 

- Binary Cross-Entropy ("BCE") : 

$$\sum_{i}^N(y_i^{true}*log(y_i^{out}) + (1 - y_i^{true})*log(1 - y_i^{out}))$$

- "Poisson" : 

$$\sum_{i}^N(y_i^{out} - y_i^{true} * log(y_i^{out}))$$
