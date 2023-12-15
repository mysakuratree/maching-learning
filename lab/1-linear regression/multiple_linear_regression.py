import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')


# 计算预测值
def compute_predicted_value(x, w, b):
    return np.dot(w, x) + b


# 计算损失
def compute_loss(f_wb, y):
    return (f_wb - y) ** 2


# 计算代价函数
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = compute_predicted_value(x[i], w, b)
        cost = cost + compute_loss(f_wb, y[i])

    cost = 1 / (2 * m) * cost

    return cost


# 计算梯度下降dw,db
def compute_gradient(x, y, w, b):
    m, n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        bias = compute_predicted_value(x[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + bias * x[i, j]
        dj_db = dj_db + bias

    dj_dw = 1 / m * dj_dw
    dj_db = 1 / m * dj_db

    return dj_dw, dj_db


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


# 加载数据集
def load_dataset():
    x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    return x_train, y_train


# 多变量线性回归
def multiple_linear_regression():
    # 加载数据集
    x_train, y_train = load_dataset()

    # 初始化
    w_init = np.zeros(x_train.shape[1])
    b_init = 0

    # 参数设置
    iterations = 1000
    alpha = 5.0e-7

    w_final, b_final, J_history, P_history = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

    # 画出代价的迭代函数
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(16, 4))
    axs.plot(J_history)
    axs.set_title("Cost vs. iteration")
    plt.show()


if __name__ == "__main__":
    multiple_linear_regression()
