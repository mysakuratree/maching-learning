# model 2 — logistic regression

### 1 The method to solve a classification problem

**1. sigmoid function**
$$
g(z) = \frac{1}{1+e^{-z}}, 0 < g(z) < 1
$$
**2. linear decision boundaries**

> 1. model

$$
z = \vec{w} \cdot \vec{x} + b
$$

**why $z = \vec{w} \cdot \vec{x} + b$ rather than $z = w_1 \cdot x_1^2 + w_2 \cdot x_2^2$? this is similar to linear regression. It's up to the training dataset, linear function is a better choice to classify the training dataset in the video**

> 2. boundary

$$
z = \vec{w} \cdot \vec{x} + b = 0
$$

> 3. logistic regression function

$$
f_{\vec{w}, b}(\vec{x}) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}
$$

**3. cost function**

> 1. loss function — make cost function convex, avoid gradient descent reaching local optimum
>    - if $y = 1$, as $f_{\vec{w}, b}(\vec{x}) \rightarrow 1$ then $loss \rightarrow 0$
>    - if $y = 1$, as $f_{\vec{w}, b}(\vec{x}) \rightarrow 0$ then $loss \rightarrow \infty$
>    - if $y = 0$, as $f_{\vec{w}, b}(\vec{x}) \rightarrow 1$ then $loss \rightarrow \infty$
>    - if $y = 0$, as $f_{\vec{w}, b}(\vec{x}) \rightarrow 0$ then $loss \rightarrow 0$

$$
L = 
\begin{cases}
-log(f_{\vec{w}, b}(\vec{x}^{(i)})) & y^{(i)} = 1 \\
-log(1-f_{\vec{w}, b}(\vec{x}^{(i)})) & y^{(i)} = 0
\end{cases}
$$

> 2. cost function

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L
$$

> 3. simplify loss function

$$
L = -y^{(i)}log(f_{\vec{w}, b}(\vec{x}^{(i)})) - (1-y^{(i)})log(1-f_{\vec{w}, b}(\vec{x}^{(i)}))
$$

> 4. simplify cost function

$$
J(\vec{w}, b) = - \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)}log(f_{\vec{w}, b}(\vec{x}^{(i)})) - (1-y^{(i)})log(1-f_{\vec{w}, b}(\vec{x}^{(i)}))]
$$

**4. gradient descent**
$$
w_i = w_i - \alpha \frac{\partial}{\partial w_i}J(\vec{w}, b)\\
b = b - \alpha \frac{\partial}{\partial b}J(\vec{w}, b)
$$

> it's suprising to find that it is similar to linear regression

$$
w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(\frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}} - y^{(i)})x_j^{(i)} \\
b = b - \alpha \frac{1}{m} \sum_{i=1}^{m}(\frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}} - y^{(i)})
$$

**5. steps**

> 1. find a suitable model
> 2. calculate loss
> 3. calculate cost function
> 4. gradient descent

### 2 underfit and overfit

**1. definition**

> 1. underfit: does not fit the training set well — high bias
> 2. overfit: fit the training set extramely well — high variance

**2. address overfitting**

> 1. collect more tarining data
> 2. decrease features(choose the most relevant features)
> 3. regularization

**3. regularization**

> 1. penalize parameter w, $\lambda$ is regularization parameter
>
> 2. the effect of $\frac{\lambda}{2m} \sum_{j=1}^{n}w_j^2$ is to keep $w_j$ small
> 3. linear regression

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n}w_j^2
$$

> gradient descent


$$
w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m}((\vec{w} \cdot \vec{x}^{(i)}+b) - y^{(i)})x_j^{(i)} + \alpha \frac{\lambda}{m}w_j \\
b = b - \alpha \frac{1}{m} \sum_{i=1}^{m}((\vec{w} \cdot \vec{x}^{(i)}+b) - y^{(i)})
$$

> 4. logisitic regression

$$
J(\vec{w}, b) = - \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)}log(f_{\vec{w}, b}(\vec{x}^{(i)})) - (1-y^{(i)})log(1-f_{\vec{w}, b}(\vec{x}^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n}w_j^2
$$

> gradient descent

$$
w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(\frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}} - y^{(i)})x_j^{(i)} + \alpha \frac{\lambda}{m}w_j \\
b = b - \alpha \frac{1}{m} \sum_{i=1}^{m}(\frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}} - y^{(i)})
$$



