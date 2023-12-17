# model 4 — decision tree

### 1 decision tree

**1. component**

> usage: classification
>
> 1. root node
> 2. decision node

**2. choose feature on each node**

> maximize purity (minimize inpurity)

**3. stop splitting**

> 1. a node is 100% on class
> 2. splitting a node will result in the tree exceeding a maximum depth
> 3. improvement in purity score are below a threshold
> 4. number of examples in a node is below a threshold

### 2 meature of impurity

> use entropy($H$) as a meature of impurity

$$
H(p) = -plog_2(p) - (1-p)log_2(1-p)\\
note: 0log0 = 0
$$

![decesion_tree](D:\VSCode\web\blog\static\article\ai\decesion_tree-1.png)

### 3 information gain

**1. definition**

> 

$$
infomation\_gain = H(p^{root}) - (w^{left}H(p^{left}) + w^{right}H(p^{right}))
$$

**2. usage**

> 1. meature the reduction in entropy
> 2.  a signal of stopping splitting

**3. continuous**

> find the threshold that has the most infomation gain

![decision_tree-2](D:\VSCode\web\blog\static\article\ai\decision_tree-2.png)

### 4 random forest

> 1. generating a tree sample

```
given training set of size m
for b = 1 to B:
	use sampling with replacement to create a new training set of size m
	train a decision tree on the training set
```

> 2. randomizing the feature choice: at each node, when choosing a feature to use to split, if n features is available, pick a random subset of k < n(usually $k = \sqrt{n}$) features and alow the algorithm to only choose from that subset of features



