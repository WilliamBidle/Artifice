# Baby-TensorFlow

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

- *sigmoid* : $\frac{1}{1 + e^{-x}}$
- *tanh* : $tanh(x)$
- *ReLU* : 

### List of avaliable loss functions:

- *MSE* : $\sum_{i}(y - \hat{y})^2$
- *MAE* : $\sum_{i}|y - \hat{y}|$
