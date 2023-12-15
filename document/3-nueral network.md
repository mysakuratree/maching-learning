# Nueral network

### 1 Demand prediction

**1. one feature**

> the activation of the price

$$
a = f(x) = \frac{1}{1+e^{-(wx+b)}}
$$

**2. mutiple features**

> 1. feature:
>
>    - features: price, shipping cost, marketing, material
>
>    - further generalization: affordability, awareness, perceived quality
>
> 2. layer: input layer, hidden layer, output layer
>
> 3. activation fuction: Sigmoid, Tanh, ReLU
>
> 4. data: transmit in the form of vector

![neural_network](D:\VSCode\web\blog\static\article\ai\neural_network-1.png)

### 2 Construct neural network layer

**1. logic implement**

> param:
>
> 1. g: sigmoid function(activation fuction)
> 2. a: activation value

![neural_network-2](D:\VSCode\web\blog\static\article\ai\neural_network-2.png)

**2. implement by tensorflow**

```
x = np.array([[...]])
y = np.array([[...]])

layer_1 = Dense(units=4, activation="sigmoid")
layer_2 = Dense(units=5, activation="sigmoid")
layer_3 = Dense(units=3, activation="sigmoid")
layer_4 = Dense(units=1, activation="sigmoid")

model = Sequential([layer_1, layer_2, layer_3, layer_4])
model.compile(...)
model.fit(x, y)

model.predict(x_new)
```





