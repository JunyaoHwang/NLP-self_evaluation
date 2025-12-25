# Gradient Blending 梯度融合



## 背景

1. 最佳的单模态网络往往比多模态的网络表现更好。
2. 1点的原因：
   1. 容量增加，多模态往往容易过拟合
   2. 不同模态过拟合和泛化的速度不同
3. 常规的避免过拟合的方式几乎没用，或者是提升十分有限。

因此提出：需要一种新的训练策略来解决多模态中联合训练的过拟合问题-->引出Gradient Blending



## OGR



作者提出：

1.我们将过拟合定义为训练集上的增益与目标分布上的增益之间的差距$O_N$：


$$
O_N = \mathcal{L}^*_N-\mathcal{L}^T_N
$$
​	其中$\mathcal{L}^*_N,\mathcal{L}^T_N$分别为在目标分布和测试集上的平均损失。





2.我们把泛化定义为对目标分布的学习量$G_N$：
$$
G_N = \mathcal L^*_N
$$


则可以定义$\Delta O_{N,n}$为在训练的第 N 个 epoch 到第 N+n 个 epoch 之间，模型“过拟合程度”增加了多少。

同理定义$\Delta G_{N,n}$为在训练的第 N 个 epoch 到第 N+n 个 epoch 之间，模型对目标分布的学习量增加了多少。也表示在训练的第 N 个 epoch 到第 N+n 个 epoch 之间，模型新获得了多少泛化能力。

将两者写为公式，有
$$
\Delta O_{N,n} = {O}_{N + n} - {O}_{N}
\\
\Delta G_{N,n} = {G}_{N + n} - {G}_{N}
$$
于是乎我们可以定义OGR:

$OGR$定义为过拟合与泛化的比率,称为过拟合-泛化比,是评判模型过拟合的指标:
$$
{OGR} \equiv  \left| \frac{\Delta {O}_{N, n}}{\Delta {G}_{N, n}}\right|  = \left| \frac{{O}_{N + n} - {O}_{N}}{{\mathcal{L}}_{N}^{ * } - {\mathcal{L}}_{N + n}^{ * }}\right|
$$
可以简单理解为模型学到冗余的信息的相对大小。很显然，我们需要最小化OGR，使得模型尽量少的过拟合。



<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250923220506679.png" alt="image-20250923220506679" style="zoom: 33%;" />

问题：

1. 我们并不知道目标分布的损失-->使用验证集上的损失作为近似

   

2. OGR计算成本极高, 难以投入实际运算

   <font size=3>代价的来源：无穷的路径选择 在拥有数百万甚至数十亿参数的高维空间中，从起点到终点有无穷多条可能的路径。在每一步更新时，梯度下降只是众多可能方向中的一种。全局优化OGR，理论上需要在一个由函数（即路径）构成的无穷维空间中进行搜索，这比在有限维参数空间中搜索要困难得多。路径的依赖性：整个轨迹的最终OGR值，并不仅仅是每一步OGR的简单累加。第 t 步的状态$θ_t$依赖于从 $θ_0 $到 $θ_{t-1} $的完整历史。这意味着，评估一条轨迹的好坏，必须从头到尾完整地模拟它，无法拆解成独立的子问题。这使得搜索空间变得难以想象的庞大和复杂，任何试图穷举或有效采样的算法都会立即面临计算上的组合爆炸.</font>

   

3. 注意到在图像左侧，当模型欠拟合的时候OGR比值也可能很小，若以OGR最小为目标函数很可能会导致模型欠拟合

对于第二个问题：

​	既然全局优化不可行，那么我们不妨采用局部优化，让这一步更新之后，带来的过拟合增长最小，而泛化能力提升最大。即最小化单步的OGR.

​	对于欠拟合的问题，前中期更多地使用SGD本身进行训练，到后期再重点使用OGR



最小化OGR等价于最小化$OGR^2$（去绝对值）

对于单次梯度更新，对其进行泰勒展开：
$$
{\mathcal{L}}^{\mathcal{T}}\left( {\theta  + \eta \widehat{g}}\right)  \approx  {\mathcal{L}}^{\mathcal{T}}\left( \theta \right)  + \eta \left\langle  {\nabla {\mathcal{L}}^{\mathcal{T}},\widehat{g}}\right\rangle
$$

$$
{\mathcal{L}}^{ * }\left( {\theta  + \eta \widehat{g}}\right)  \approx  {\mathcal{L}}^{ * }\left( \theta \right)  + \eta \left\langle  {\nabla {\mathcal{L}}^{ * },\widehat{g}}\right\rangle
$$


$$
{OG}{R}^{2} = {\left( \frac{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}}\left( {\theta }^{\left( i\right) }\right)  - \nabla {\mathcal{L}}^{ * }\left( {\theta }^{\left( i\right) }\right) ,{\widehat{g}}_{i}\right\rangle  }{\left\langle  \nabla {\mathcal{L}}^{ * }\left( {\theta }^{\left( i\right) }\right) ,{\widehat{g}}_{i}\right\rangle  }\right) }^{2}
$$
其中，梯度$\hat{g}$由多个模态的梯度加权得到。对于M个模态，第k个模态的梯度为$v_k$,其权重为$w_k$。所以$OGR^2$可以进一步写为
$$
{OG}{R}^{2} = {\left( \frac{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}}\left( {\theta }^{\left( i\right) }\right)  - \nabla {\mathcal{L}}^{ * }\left( {\theta }^{\left( i\right) }\right) ,\sum w_kv_k\right\rangle  }{\left\langle  \nabla {\mathcal{L}}^{ * }\left( {\theta }^{\left( i\right) }\right) ,\sum w_kv_k\right\rangle  }\right) }^{2}
$$
而我们需要做的是找到$w^*$最小化OGR.即：
$$
{w}^{ * } = \underset{w}{\arg \min }\mathbb{E}\left\lbrack  {\left( \frac{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  }{\left\langle  \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  }\right) }^{2}\right\rbrack
$$
在OGR的公式中，我们不难发现，OGR的大小是不会随着$\tilde{g}$的变化而发生改变的。

所以为了方便计算，我们不妨规定$\tilde{g}$的大小，使得：
$$
{\left\langle  \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle } = 1
$$
所以原问题可化简为
$$
{w}^{ * } = \underset{w}{\arg \min }\mathbb{E}\left\lbrack  {{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  } }^{2}\right\rbrack\\= \underset{w}{\arg \min }\mathbb{E}\left\lbrack  {{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  } * {\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  }}\right\rbrack
$$




<span style='color:red'>关键假设：不同梯度估计的过拟合项是不相关的，也就是说，当$i\neq j$时，有:</span>
$$
\mathbb{E}\left\lbrack  {\left\langle  {\nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{k}}\right\rangle  \left\langle  {\nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{j}}\right\rangle  }\right\rbrack   = 0
$$




对$\mathbb{E}\left\lbrack  {{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  } * {\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle  }}\right\rbrack$进一步展开：
$$
= \mathbb{E}\left\lbrack  {\sum\limits_{k,j}w_kw_j{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  } * {\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{j}\right\rangle  }}\right\rbrack\\=\sum\limits_{k,j}w_kw_j\mathbb{E}\left\lbrack  {{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  } * {\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{j}\right\rangle  }}\right\rbrack
$$
注意到在假设下当且仅当k=j时求和中的单项不为0

所以
$$
{w}^{ * } =\sum\limits_{k}w_k^2\mathbb{E} {{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  }^2}
$$
令$\sigma_k^2 = \mathbb{E} {{\left\langle  \nabla {\mathcal{L}}^{\mathcal{T}} - \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  }^2}$

所以$w^*$进一步化简为
$$
{w}^{ * } = \underset{w}{\arg \min }\sum_{k}w_k^2\sigma_k^2
$$

让我们重新明确优化目标和约束条件：
$$
{w}^{ * } = \underset{w}{\arg \min }\sum_{k}w_k^2\sigma_k^2\\
s.t.\quad  {\left\langle  \nabla {\mathcal{L}}^{ * },\mathop{\sum }\limits_{k}{w}_{k}{v}_{k}\right\rangle } = 1
$$
使用拉格朗日：
$$
L = \mathop{\sum }\limits_{k}{w}_{k}^{2}{\sigma }_{k}^{2} - \lambda \left({\mathop{\sum }\limits_{k}{w}_{k}\left\langle  {\nabla {\mathcal{L}}^{ * },{v}_{k}}\right\rangle   - 1}\right)
$$
$对L求w_k求偏导：$
$$
\frac{\partial L}{\partial w_k} = 2w_k\sigma_k^2-\lambda\langle\nabla L^*,v_k\rangle
$$
偏导为0解得：
$$
w_k = \frac{\lambda\langle\nabla L^*,v_k\rangle }{2\sigma_k^2}
$$


将这个解带入到约束目标：
$$
1 = \mathop{\sum }\limits_{k}{w}_{k}\left\langle  {\nabla {\mathcal{L}}^{ * },{v}_{k}}\right\rangle   = \lambda \mathop{\sum }\limits_{k}\frac{{\left\langle  \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  }^{2}}{2{\sigma }_{k}^{2}}
$$
解得$\lambda$:
$$
\lambda  = \frac{2}{\mathop{\sum }\limits_{k}\frac{{\left\langle  \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  }^{2}}{{\sigma }_{k}^{2}}}
$$
令$Z = 1/\lambda$,得到${w}_{k}^{ * } = \frac{1}{Z}\frac{{\left\langle  \nabla {\mathcal{L}}^{ * },{v}_{k}\right\rangle  }^{2}}{2{\sigma }_{k}^{2}}$。再除以权重之和，得到归一化权重。

