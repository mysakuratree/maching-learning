# model 1 — linear regression

### 1 The method to find suitable parameters via cost function

**1. steps**

> 1. find a suitable model
> 2. calculate cost function
> 3. min cost function — gradient descent

**2. gradient descent**

> take model $f(x) = wx + b$ as an example
>
> 1. cost function: 

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m}((wx^{(i)}+b) - y^{(i)})^2
$$

> 2. gradient descent
>    - $\alpha$ is learning rate
>    - $J(w, b)$ is cost function

$$
w = w - \alpha \frac{\partial}{\partial w}J(w, b)\\
b = b - \alpha \frac{\partial}{\partial b}J(w, b)
$$

**3. notice**

> if implement with code, make sure w and b are updated from origin w and b

$$
tmp\_w = w - \alpha \frac{\partial}{\partial w}J(w, b)\\
tmp\_b = b - \alpha \frac{\partial}{\partial b}J(w, b)\\
w = tmp\_w\\
b = tmp\_b
$$

**4. learning rate**

> it's a rather important point to chose a suitable learning rate
>
> 1. $\alpha$ is too small: convergence is too slow
> 2. $\alpha$ is too large: overshoot

**5. gradient descent for multiple variables**

> take $f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$ as an example
>
> 1. cost function

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x}) - y^{(i)})^2
$$

> 2. gradient descent

$$
w_i = w_i - \alpha \frac{\partial}{\partial w_i}J(\vec{w}, b)\\
b = b - \alpha \frac{\partial}{\partial b}J(\vec{w}, b)
$$

> 3. formula for code

$$
w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m}((\vec{w} \cdot \vec{x}^{(i)}+b) - y^{(i)})x_j^{(i)} \\
b = b - \alpha \frac{1}{m} \sum_{i=1}^{m}((\vec{w} \cdot \vec{x}^{(i)}+b) - y^{(i)})
$$

### 2 Accelerate convergence through feature scaling

> take $300 \le x_1 \le 200, 0 \le x_2 \le 5$ as an example

**1. max normalization**
$$
x_{1, scaling} = \frac{x_1}{2000}\\
x_{2, scaling} = \frac{x_2}{5}
$$
**2. mean normalization**

> $\mu$ is average

$$
x_{1, scaling} = \frac{x_1 - \mu_1}{2000-300}\\
x_{2, scaling} = \frac{x_2 - \mu_2}{5-0}
$$

**3. Z-score normalization**

> transformed into a standard normal distribution
>
> $\mu$ is average
>
> $\sigma$ is standard deviation

$$
x_{1, scaling} = \frac{x_1 - \mu_1}{\sigma_1}\\
x_{2, scaling} = \frac{x_2 - \mu_2}{\sigma_2}
$$

### 3 Tips for linear regression

> 1. plot learning curve
> 2. set a suitable learning rate, try ··· 0.001, 0.003, 0.01, 0.03 ···
> 3. conduct feature engineering, choose suitable feature



