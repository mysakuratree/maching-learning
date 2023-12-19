# model 4 — K-means

### 1 definition

> 1. randomly initialize K cluster centroids $\mu_1, \mu_2, \cdots$
> 2. repeat:
>    - assign each point to its closest centroid $\mu$
>    - recompute the centroids(average of the closest point)

![k-means](D:\VSCode\web\blog\static\article\ai\k-means.png)

### 2 optimazation objective

> 1. $c^{(i)}$ = index of cluster to which example $x^{(i)}$ is currently assigned
> 2. $\mu_k$ = cluster centroid k
> 3. $\mu_{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned

$$
J = \frac{1}{m} \sum_{i=1}^m \| x^{(i)} - \mu_{c^{(i)}} \|
$$

### 3 randomly initialization

```
for i = 1 to n(usually 50 to 1000)
	randomly initialize K-means
	run K-means
	compute cost function
	
pick set of clusters that give the lowest cost
```



