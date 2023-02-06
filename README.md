# Baby-TensorFlow

A homemade machine learning neural network architecture modeled after TensorFlow.

To install Baby_Tensorflow, use the command:

    pip install git+https://github.com/WilliamBidle/Baby-TensorFlow

The full list of functions, their usage, as well as some examples can be found within the ***Baby_Tensorflow.py*** file.

List of available activation functions:

- *sigmoid* : $\frac{1}{1 + e^{-x}}$
- *tanh* : $tanh(x)$
- *ReLU* : $Y(i,k) = 
\left\{
    \begin{array}{lr}
        ||R_{k}-R_{i}||^{2}, & \text{if } i \neq k\\
        ||\triangle_{i}||^{2}, & \text{if } i\leq k
    \end{array}
\right\} = yz$

List of avaliable loss functions:

-
-
