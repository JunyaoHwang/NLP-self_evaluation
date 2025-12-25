# ARL论文讲解

## 1 研究背景

​	现有的多模态的研究存在着欠优化的问题。现有的方法，例如OGM-GE将其归咎于模态之间的不平衡学习。因为表现好的模态往往会抑制表现差的模态的更新。因此希望通过平衡梯度的方式解决。这些论文中，以OGM-GE为例，认为

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919153538826.png" alt="image-20250919153538826" style="zoom: 50%;" />

梯度要尽可能的平衡，其中默认不同模态应该在梯度更新中有着相等的“话语权”。这样做显然忽视了模态之间的表现的差异，即有些模态实际上对多模态的预测确实有着更大的贡献，而有的模态本身的贡献却没这么大。在这种情况下去平衡梯度的做法是有待商榷的。

OGM-GE中的差异比例设定：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919154924914.png" alt="image-20250919154924914" style="zoom:67%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919155623854.png" alt="image-20250919155623854" style="zoom: 80%;" />



可以看到，OGM-GE中，其制定了这个梯度调制的参数k<sub>t</sub>，当ρ大于1时，即视频在正确分类上的输出概率大于音频在正确分类上的输出概率时，我们需要削弱video的梯度的权重。防止video主导训练造成不平衡学习。

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919160657189.png" alt="image-20250919160657189" style="zoom: 50%;" />

我们不难看出，它们认为所有模态同等重要，忽略了模态之间固有的能力差异。

## 2 作者观点

平衡优化并非多模态学习的最佳设置；相反，遵循模态方差的不平衡学习可以带来更好的性能。

## 3 与方差呈反比

作者发现，把表现较好的模态的梯度增大 5 倍、加剧训练不平衡时，模型性能并未下降反而提升。证明平衡学习并非最优

作者以vanilla拼接模型为例:

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919161928261.png" alt="image-20250919161928261" style="zoom: 50%;" />

分别为优化目标：最小化泛化误差  和  f(x<sub>i</sub>)表达式。

拆分g为Bias,Var和不可约误差。

因此，为了最小化泛化误差，我们可以调整w<sub>0</sub>和w<sub>1</sub>

尝试让Bias或者Var为0。

### 尝试使Bias为0

分解Bias:

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919162813886.png" alt="image-20250919162813886" style="zoom:25%;" />

令Bias为0，解得：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919162717802.png" alt="image-20250919162717802" style="zoom: 25%;" />

我们发现，在这组解中w0和w1必定异号，显然不成立。

即：找不到一组(w0,w1)，消掉误差中的偏差项

### 尝试使Var为0

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919163039976.png" alt="image-20250919163039976" style="zoom:25%;" />

我们发现，当Var为0时，w0/w1 = Var1/Var0 即按照方差的倒数来分配权重可以消除误差中的方差项

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919163349672.png" alt="image-20250919163349672" style="zoom:25%;" />



即：让方差更小的模态贡献更多



## 4 方差的近似和梯度更新



偏差方差分解是为传统回归问题设计的，对于分类问题而言，由于输出层是 softmax 概率分布(或者logits),难以计算方差。同时，熵和方差都是作为不确定性的度量，所以使用熵来代替方差。（论文：Neha Gupta, Jamie Smith, Ben Adlam, and Zelda E Mariet. Ensembles of classifiers: a bias-variance perspective. Trans-actions on Machine Learning Research, 2022.）

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919164402754.png" alt="image-20250919164402754" style="zoom:25%;" />

计算q^{m_0}/q^{m_1},得到ARL所得出的理想权重比例。

将这个理想权重比和实际值(d的比值)做softmax得出对应的梯度增益的系数
$$
[a_{m0}, a_{m1} ] = σ[\frac{q_{m0}}{q_{m1}} T,\frac{d_{m0}}{d_{m1}}T]
$$
当模态m0学习不足，即d比值<q比值，则会给予m0更高的梯度的权重

当模态m0学习过度，即d比值>q比值，则会给予m1更高的梯度的权重

更新梯度:

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919173424339.png" alt="image-20250919173424339" style="zoom: 50%;" />

其中，在a*g的基础上加上了g_s是为了当a极小时参数停止更新。

## 5 对于偏差的处理

我们发现目前为止没有专门针对偏差的优化。为了减少偏差，引入单模态损失函数：

将其他模态遮住(输入置为0)，只剩下该模态之后计算出的损失。

各个模态计算各自的损失函数之后相加并乘上系数 γ，得到单模态损失部分，添加到损失函数中。

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250919174036442.png" alt="image-20250919174036442" style="zoom:25%;" />



## 6 总结

本文提出了以方差倒数为比例进行不平衡学习，以这样的权重进行学习可以有效降低泛化误差的方差项。由于无法找到合适的权重来降低偏差，所以作者添加了单模态损失函数进行优化来弥补偏差部分。



## 7 优点缺点

提出ARL，从偏差方差分解的角度出发，成效显著

方法简单高效，容易嵌入，通用性强

**本文相较于OMG-GE等平衡学习方法，由于其本身使用了方差参与到整体的计算中，可以有效防止方差过小和过大，替代了KL散度正则项。**

缺点：

由于偏差的存在，导致按照方差反比进行不平衡学习未必会有好的效果

ARL高度依赖超参T和$γ$的影响。调参不方便。

ARL 选择直接优化各个单模态的 bias，但这种处理方式 仍然没有保证最终融合 bias 的全局最优性，只是间接缓解。

ARL 的理论推导基于 模态方差的独立性 假设，只用单模态方差来分配权重，忽视了模态之间的协同和冗余。同时，模态和模态之间往往具有一定的相关性。



## 8 潜在的可改进的点

本文同样给予各个模态的单模态损失函数相同的权重(均未乘以权重系数)，可以在此基础上进一步改进

本文把方差和偏差分开计算，我们是否可以找到合适的W，对方差和偏差进行整体优化？

单模态的损失计算可否优化
