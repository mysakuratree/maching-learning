import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')


# 计算代价
def compute_cost(x, y, w, b):
    # 获取数据条数
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2

    cost = 1 / (2 * m) * cost

    return cost


# 计算dw、db
def compute_gradient(x, y, w, b):
    # 获取数据条数
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw = dj_dw + (f_wb - y[i]) * x[i]
        dj_db = dj_db + (f_wb - y[i])

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

        print(f"第{i}次迭代：w：{w: 0.3e}，b：{b: 0.3e}")

    return w, b, J_history, P_history


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

    # 画出代价的迭代函数
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(16, 4))
    axs.plot(J_history)
    axs.set_title("Cost vs. iteration")
    plt.show()


if __name__ == '__main__':
    linear_regression()
