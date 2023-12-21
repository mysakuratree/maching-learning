# Reinforcement learning

### 1 key concepts

> 1. states
> 2. actions
> 3. rewards
> 4. discount factor $\gamma$
> 5. return
> 6. policy $\pi$

### 2 return

> 1. definition: the sum of the rewards that the system gets, weighted by the discount factor
>
> 2. compute:
>    - $R_i$ : reward of state i
>    - $\gamma$ : discount factor(usually close to 1), making the reinforcement learning impatient

$$
return = R_1 + \gamma R_2 + \cdots + \gamma^{n-1} R_n
$$

### 3 policy

> policy $\pi$ maps state $s$ to some action $a$

$$
\pi(s) = a
$$

> the goal of reinforcement learning is to find a policy $\pi$ to map every state $s$ to action $a$ to maximize the return

![reinforcement_learning-1](D:\VSCode\web\blog\static\article\ai\reinforcement_learning-1.png)

### 4 state action value function

**1. definition**

> $Q(s, a) = $return if
>
> - start in state $s$
> - take action $a$ once
> - behave optimally after that

**2. usage**

> 1. the best possible return from state $s$ is $max$ $Q(s, a)$
> 2. the best possible action in state $s$ is the action $a$ that gives $max$ $Q(s, a)$

### 5 bellman equation

> $s$ : current state
>
> $a$ : current action
>
> $s^{'}$ : state you get to after taking action $a$
>
> $a^{'}$ : action that you take in state $s^{'}$

$$
Q(s, a) = R(s) + \gamma max Q(s^{'}, a^{'})
$$

### 6 Deep Q-Network

**1. definition**

> use neural network to learn $Q(s, a)$

$$
x = (s, a)\\
y = R(s) + \gamma max Q(s^{'}, a^{'}) \\
f_{w, b}(x) \approx y
$$

![reinforcement_learning-2](D:\VSCode\web\blog\static\article\ai\reinforcement_learning-2.png)

**2. step**

> 1. initialize neural network randomly as guess of $Q(s, a)$
> 2. repeat:
>    - take actions, get $(s, a, R(s), s^{'})$
>    - store N most recent $(s, a, R(s), s^{'})$ tuples
> 3. train neural network:
>    - create training set of N examples using $x = (s, a)$ and $y = R(s) + \gamma max Q(s^{'}, a^{'})$
>    - train $Q_{new}$ such that $Q_{new} \approx y$
>    - set $Q = Q_{new}$

**3. optimazation**

![reinforcement_network-3](D:\VSCode\web\blog\static\article\ai\reinforcement_network-3.png)

**4. $\epsilon$-greedy policy**

> 1. with probability $1 - \epsilon$, pick the action $a$ that maximize $Q(s, a)$
> 2. with probability $\epsilon$, pick the action $a$ randomly

**5. mini-batch**

> use a subset of the dataset on each gradient decent

**6. soft update**

> instead $Q = Q_{new}$

$$
w = \alpha w_{new} + w\\
b = \alpha b_{new} + b
$$



