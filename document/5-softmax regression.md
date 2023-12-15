# model 3 - softmax regression

### 1 Multiclass classification problem

**1. softmax function**

$$
z_i = \vec{w_i} \cdot \vec{x} + b_i \\
a_i = g(z_1 \cdots z_n) = \frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
$$
**2. loss function**
$$
L = -log(a_i) \quad y = i
$$

**3. cost function**
$$
J = \frac{1}{m} \sum_{i=1}^{m} L
$$

### 2 Construct neural network

**1. logic implement**

![softmax_regression](D:\VSCode\web\blog\static\article\ai\softmax_regression.png)

**2. implement by code**

```
x = np.array([[...]])
y = np.array([[...]])

layer_1 = Dense(units=25, activation="relu")
layer_2 = Dense(units=15, activation="relu")
layer_3 = Dense(units=10, activation="softmax")

model = Sequential([layer_1, layer_2, layer_3])
model.compile(loss=SparseCategoricalCrossentropy())
model.fit(x, y, epochs=100)

model.predict(x_new)
```

**3. Improved implement**

> increase the accuracy of the computing bu omitting some procedure variables

```
x = np.array([[...]])
y = np.array([[...]])

layer_1 = Dense(units=25, activation="relu")
layer_2 = Dense(units=15, activation="relu")
layer_3 = Dense(units=1, activation="linear")

model = Sequential([layer_1, layer_2, layer_3])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(x, y, epochs=100)

model.predict(x_new)
```

**4. Adam**

```
x = np.array([[...]])
y = np.array([[...]])

layer_1 = Dense(units=25, activation="relu")
layer_2 = Dense(units=15, activation="relu")
layer_3 = Dense(units=10, activation="linear")

model = Sequential([layer_1, layer_2, layer_3])
model.compile(
				optimizer=Adam(learning_rate=1e-3),
				loss=SparseCategoricalCrossentropy(from_logits=True)
				)
model.fit(x, y, epochs=100)

model.predict(x_new)
```




