# logistic regression模型机器学习实战

### 1 功能模块

**1. 功能函数**

> 1. sigmoid

```
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

> 2. f_wb

```
# 计算预测值
def compute_predicted_value(x, w, b):
    return np.dot(w, x) + b
```

**2. 损失函数**

> 照着公式写出来

```
# 计算损失
def compute_loss(f_wb, y):
    return -y * np.log2(f_wb) - (1 - y) * np.log2(1 - f_wb)
```

**3. 代价函数**

```
# 计算代价
def compute_cost(x, y, w, b):
    m, n = x.shape
    cost = 0
    for i in range(m):
        z = compute_predicted_value(x[i], w, b)
        f_wb = sigmoid(z)
        cost = cost + compute_loss(f_wb, y[i])

    cost = -1 / m * cost

    return cost
```

**4. 计算梯度下降值**

```
# 计算梯度下降值dw，db
def compute_gradient(x, y, w, b):
    m, n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        z = compute_predicted_value(x[i], w, b)
        bias = sigmoid(z) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + bias * x[i, j]
        dj_db = dj_db + bias

    dj_dw = 1 / m * dj_dw
    dj_db = 1 / m * dj_db

    return dj_dw, dj_db
```

**5. 进行梯度下降**

```
# 梯度下降
def gradient_descent(x, y, w, b, alpha, num_iters):
    # 代价函数与w、b的历史副本
    J_history = []
    P_history = []

    for i in range(num_iters):
        # 计算梯度下降值
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # 更新两个参数
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(compute_cost(x, y, w, b))
        P_history.append((w, b))

        print(f"第{i}次迭代：w：{w}，b：{b}")

    return w, b, J_history, P_history
```

**6. 加载数据集**

> 本次还是简略设置一个即可

```
# 加载数据集
def load_dataset():
    x_train = np.array([[1, 0.5], [0.5, 1], [0.5, 0.5], [2, 2], [2.5, 2], [2, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    return x_train, y_train
```

**7. 可视化函数**

> 可视化结果

```
# 画出代价的迭代函数和最终的决策边界
def visual(x_train, y_train, w_final, b_final, J_history):
    fig, (axs1, axs2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    axs1.plot(J_history)
    axs1.set_title("Cost vs. iteration")

    plot_data(x_train, y_train, axs2)
    x0 = -b_final / w_final[1]
    x1 = -b_final / w_final[0]
    axs2.plot([0, x0], [x1, 0], c=dlc["dlblue"], lw=1)
    axs2.set_title("Decision boundary")

    plt.show()
```

**8. 进行逻辑回归**

```
# 逻辑回归
def logistic_regression():
    # 加载数据集
    x_train, y_train = load_dataset()

    # 初始化
    w_init = np.zeros(x_train.shape[1])
    b_init = 0

    # 参数设置
    iterations = 5000
    alpha = 0.1

    w_final, b_final, J_history, P_history = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

    visual(x_train, y_train, w_final, b_final, J_history)
```

### 2 结果展示

> 直接上图，看得出来，本次实验很成功

![logistic_regression](D:\VSCode\web\blog\static\article\ai\logistic_regression.png)



