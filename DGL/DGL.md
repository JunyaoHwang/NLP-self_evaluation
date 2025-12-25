# DGL

## 现状

在多模态的训练中，我们期望结合多个模态的信息源的模型比根据单个模态的信息源的模型的预测效果更好。但是真实情况是模态融合的预测效果相较于最好的单模态不升反降。以往的方法认为这是模态之间的不平衡学习导致的，但是这些方法忽视了: 为什么主导的模态在多模态训练中的性能比它单模态训练的性能要差？

## 作者的发现

多模态模型中模态编码器和融合模块之间编码冲突-->多模态模型中所有模态未得到充分优化



## 解决办法

让编码器使用该模态自身的损失函数进行优化，让编码之后的阶段使用联合训练的损失函数进行优化



## 详解

以两个模态的情况为例：

对于训练集$\mathcal{D}=\{x_i,y_i\}_{i\in [N]}$

样本$x$：$x_i = (x^{m_1},x^{m_2})$ 一个样本由两个模态的数据组成

类别$y$：$y_i \in [1,2,\dots,K]$ 一个样本属于K类中的一个

编码器：$\varphi_1(\theta_1,\cdot)，\varphi_2(\theta_2,\cdot)$

特征向量$z_i^{m_1} = \varphi_1(\theta_1,x_i^{m_1}),z_i^{m_2} = \varphi_i(\theta_2,x_i^{m_2})$

融合模块$\varphi_\tau$
$$
\begin{cases}
L^{multi} = L_{CE}(Wz_i^\tau + b,y_i) \\
z_i^\tau = \varphi_\tau(\theta_\tau,z_i^{m_1},z_i^{m_2})
\end{cases}
$$

回传给编码器$\varphi_1$的梯度是$\frac{\partial L^{Multi}}{\partial z_i^{m_1}}$

记$f(z_i^{\tau}) = Wz_i^\tau+b$

根据链式法则写成如下形式:
$$
g^{\mathrm{Multi}}_{\theta_1}
= \frac{\partial L^{\mathrm{Multi}}}{\partial f(z_i^{\tau})}
  \frac{\partial f(z_i^{\tau})}
 {\partial z_i^{m_1}}
$$
假设融合模块设置为拼接，即$z_i^\tau = [z_i^{m_1},z_i^{m_2}]$

则可知：
$$
f(z_i^\tau) = W_1z_i^{m_1}+W_2z_i^{m_2}+b
\\
\frac{\partial{f(z_i^\tau)}}{\partial{z_i^{m_1}}} = W_1
$$
推导$\frac{\partial L}{\partial f}$：
$$
L = \sum_{i=1}^k y_k \log(p_k)
$$
y是one-hot编码的
$$
L = -y_i\log(p_i)=-\log(p_i)
$$
对p求导
$$
\frac{\partial L}{\partial p} = -\frac{1}{p_i}\\
$$
$p_i$对$f_i$的导数
$$
\frac{\partial p_i}{\partial f_j} =\begin{cases}{p_i(1-p_i)} \quad i=j\\-p_ip_j\quad i≠j\end{cases}
$$



$$
p_{y_i}^{Multi} = \frac{e^{(W_{y_i}^{m_1} z_i^{m_1}+b + W_{y_i}^{m_2} z_i^{m_2})}}{\sum_{k=1}^{K} e^{(W_k^{m_1} z_i^{m_1}+b + W_k^{m_2} z_i^{m_2})}}
$$

$$
p_{y_i}^{Multi} = \frac{e^{(W_{y_i}^{m_1} z_i^{m_1} +b+ W_{y_i}^{m_2} z_i^{m_2})}}{\sum_{k=1}^{K} e^{(W_k^{m_1} z_i^{m_1} +b+ W_{y_i}^{m_2} z_i^{m_2})} \cdot e^{(W_k^{m_2} - W_{y_i}^{m_2}) z_i^{m_2}}}
$$

$$
 p_{y_i}^{Multi} = \frac{e^{(W_{y_i}^{m_1} z_i^{m_1})}}{\sum_{k=1}^{K}  e^{(W_k^{m_1} z_i^{m_1})} \cdot e^{(W_k^{m_2} - W_{y_i}^{m_2})  z_i^{m_2}}} 
 
$$

所以L对$g_{\theta_1}^{multi}$求导得到
$$
g_{\theta_1}^{Multi} = \left( \frac{e^{(W_{y_i}^{m_1} z_i^{m_1} + b_1)}}{\sum_{k=1}^{K} e^{(W_k^{m_1} z_i^{m_1} + b_1)} e^{(W_k^{m_2} - W_{y_i}^{m_2}) z_i^{m_2}}} - 1 \right) W_{y_i}^{m_1}
$$
单模态损失：
$$
g_{\theta_1}^{Uni} = \left( \frac{e^{(W_{y_i}^{m_1} z_i^{m_1} + b_1)}}{\sum_{k=1}^{K} e^{(W_k^{m_1} z_1^{m_1} + b_1)}} - 1 \right) W_{y_i}^{m_1}
$$
我们已知
$$
\begin{cases} 
e^{(W_k^{m_2} - W_{y_i}^{m_2}) z_i^{m_2}} < 1, & \text{if } k \neq y_i \\
e^{(W_k^{m_2} - W_{y_i}^{m_2}) z_i^{m_2}} = 1, & \text{if } k = y_i 
\end{cases}
$$
我们可以将权重矩阵 W 的第 k行向量 $W_k$ 理解为模型学习到的第k个类的‘模板’或‘决策边界法向量’。分类的过程，就是计算输入样本的特征向量 z与哪一个类的‘模板’ $W_k$点积最大，z在哪个 $W_k $方向上的投影最长）。这个点积$W_k · z$`直接构成了送入Softmax的logit。

模型应该学习到，使得对于模态2而言，它给正确类别打出的分数，要高于它给任何一个错误类别打出的分数。

于是我们发现：$g^{multi}_{\theta_1}$相较于$g_{\theta_2}^{uni}$而言，其括号内部分母更小，导致分式值变大但始终小于1，导致:
$$
\text{abs}\left(g_{\theta_1}^{Uni}\right) > \text{abs}\left(g_{\theta_1}^{Multi}\right) > 0
$$
所以问题出现了：使用联合训练的损失函数对编码器进行更新会削弱编码器的梯度更新幅度。

如果$m_2$是易学习的模态，其$e^{(W_k^{m_2} - W_{y_i}^{m_2}) z_i^{m_2}}<e^{(W_k^{m_1} - W_{y_i}^{m_1}) z_i^{m_1}}$更大，说明易于学习的模态引起的梯度抑制将大于难以学习的模态。这解释了为什么当模态梯度相互干扰时，易于学习的模态比难以学习的模态表现更好。

所以，如果我们想要解决这个问题，那么需要$e^{(W_k^{m_2} - W_{y_i}^{m_2}) z_i^{m_2}}=1$，所以需要$z_i^{m_2}=0$,意味着融合模块完全忽略了模态2的信息。



算法流程

```
    输入：训练数据集 D，迭代次数 T，超参数 - 参数 α。
    对于 t = 0, ..., T - 1 执行
        将批处理数据前馈到模型中。
        通过计算单峰损失Lm1和Lm2。
        计算单峰梯度。
        消除从……传回的梯度。
        融合模块的单峰损失。
        使用 detach 计算多模态损失 Ld。
        通过公式 10 表示。
        计算多模态梯度。
        更新模型参数。
```

单峰损失函数：
$$
\begin{cases} 
L^{m_1} = L_{CE} \left( W \left( z_i^{\tau 1} + b, y_i \right) \right) \\
z_i^{\tau 1} = \varphi_\tau \left( \theta_\tau, z_i^{m_1}, 0 \right)
\end{cases}
$$






## 我的构想

由于冗余信息在多个模态中都存在，可以进行一定程度上的互补。所以模态独特信息和协同信息的缺失可能会给模型造成更大程度的影响。为了更好地提取模态的独特信息，我在编码器中添加PID信息论的损失函数。设计提取独特信息的专家并设计损失函数$L^{uni}$，给予每个模态的编码器一个自适应的参数$\beta$，则新的用于更新编码器损失函数为
$$
L^{new}_i = L^{original}_i + \beta L^{uni}_i
$$
而优势模态的独特信息往往能够被较好地提取，所以这个方法主要针对弱模态的独特信息的提取。$\beta$为一个自适应的参数。模态弱-->$\beta$更大，允许模态更多提取独特信息。
