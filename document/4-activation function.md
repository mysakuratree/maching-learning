# Activation function

### 1 Reason

> 1. introduction of Non-linearity
> 2. making gradient descent possible
> 3. providing a basis for decision-making

### 2 Activation function classification

**1. sigmoid**

> maps any real number to the range (0,1)

$$
g(z) = \frac{1}{1+e^{-z}}
$$

**2. ReLU**

> currently the most commonly used activation function, helps alleviate the vanishing gradient problem during backpropagation

$$
g(z) = max(0, z)
$$

**3. Tanh**

> maps any real number to the range (-1,1)

$$
g(z) = \frac{sinh(z)}{cosh(z)} = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

**4. softmax**

> maps any real number to a probability distribution within the range (0,1), often used in the output layer of multi-classification neural networks

$$
z_i = \vec{w_i} \cdot \vec{x} + b_i \\
g(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
$$

### 3 The choice of activation function

**1. output layer**

> 1. binary classification: sigmoid
> 2. multiple classification: softmax
> 3. regression: 
>    - y neg or pos: linear
>    - y pos: ReLU

**2. hidden layer**

> most commonly used: ReLU



