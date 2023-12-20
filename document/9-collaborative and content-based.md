# Collaborative and Content-based filter

### 1 collaborative filter

**1. definition**

> recommend items based on rating of users who gave similar rating

**2. cost function**

> 1. learn $w^{(1)}, b^{(1)}, \cdots, w^{(n_u)}, b^{(n_u)}$

$$
J = \frac{1}{2}\sum_{j=1}^{n_u} \sum_{i:r(i,j) = 1}(w^{(j)} \cdot x^{(i)} - b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(w_k^{(j)})^2
$$

> 2. learn $x^{(1)}, \cdots, x^{(n_m)}$

$$
J = \frac{1}{2}\sum_{j=1}^{n_u} \sum_{i:r(i,j) = 1}(w^{(j)} \cdot x^{(i)} - b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(x_k^{(i)})^2
$$

> 3. collaborative filter

$$
J(w, b, x) = \frac{1}{2}\sum_{j=1}^{n_u} \sum_{i:r(i,j) = 1}(w^{(j)} \cdot x^{(i)} - b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(w_k^{(j)})^2 + \sum_{j=1}^{n_u} \sum_{k=1}^{n}(x_k^{(i)})^2
$$

> 4. for binary labels
>
> $$
> g(x) = \frac{1}{1+e^{-z}}\\
> f(x) = g(w^{(j)} \cdot x^{(i)} - b^{(j))})\\
> L = -y^{(i, j)}log(f(x)) - (1 - y^{(i, j)})log(1-f(x))\\
> j(w, b, x) = \sum L
> $$
>
> 

### 2 Content-based filter

**1. definition**

> recommand items based on fearures of user and item to find good match

**2. cost function**
$$
J = \sum_{(i, j):r(i, j)=1}(v_u^{(j)}v_m^{(j)} - y^{(i, j)})^2 + NN(regularization-term)
$$
![collaborative](D:\VSCode\web\blog\static\article\ai\collaborative.png)



