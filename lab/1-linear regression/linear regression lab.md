# linear regression模型机器学习实战

**1. 代价函数**

```
# 计算代价
def compute_cost(x, y, w, b):

    # 获取数据条数
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2

    cost = 1 / (2 * m) * cost

    return cost
```

**2. 梯度下降变化值**

```
# 计算dw、db
def compute_gradient(x, y, w, b):

    # 获取数据条数
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])

    dj_dw = 1 / m * dj_dw
    dj_db = 1 / m * dj_db

    return dj_dw, dj_db
```

**3. 写出进行梯度下降的主函数**

```
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
        P_history.append([w, b])

        print(f"第{i}次迭代：w：{w: 0.3e}，b：{b: 0.3e}")

    return w, b, J_history, P_history
```

**4. 导入数据集进行迭代**

> 这里数据集进行了简化，但可更换真实数据集

```
def linear_regression():

    # 数据集
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    # 初始化
    w_init = 0
    b_init = 0

    # 参数设置
    iterations = 5000
    alpha = 1.0e-2

    w_final, b_final, J_history, P_history = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

    # 这里借助吴恩达提供的画图函数，把迭代函数画出来
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(16,4))
    axs.plot(J_history)
    axs.set_title("Cost vs. iteration")
    plt.show()
```

**5. 迭代结果**

> 其实这里有迭代图，懒得展示了，感兴趣可以复制下来运行一下linear_regression()这个函数看看

```
第4999次迭代：w： 1.997e+02，b： 1.004e+02
```

> 与预期的w=2.0，b=1差别不大，成功



