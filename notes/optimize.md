## Optimize

### 关于梯度下降法
重要问题出现在：grad=0处未必是最小值点
在讨论这个问题前，需要讨论另一个问题：**grad=0未必是极小值点**

> 极小值点与鞍点：仅仅有grad还不足以区分，需要计算Loss Function的Hessian矩阵，正定的H意味着该点真的是极小值点，半正定说明只是一个鞍点。

> Insight：事实上，一个一维的local minima在二维中很可能是一个鞍点，所以，在大量参数构成的高维像空间中，大多数我们以为的local minima很可能只是一个鞍点。

---

### Momentum

思想是我在考虑这一次update时也加入上一次移动的步长（像是一种惯性）

比如说我不带momentum就是：$m^i=-\eta\cdot grad^i$
带着momentum就是：$m^i=-\eta\cdot grad^i+\lambda\cdot m^{i-1}$

---

### Adam

动机是希望learning rate可以有一个自动的调整

> 朴素的想法 **RMS**：$m_j^i=-{{\eta}\over{\sigma ^i_j}}grad^i_j,\sigma ^i_j$是**j分量上的**截止到第i步的i个步长的方均根
> 这种想法的作用是，遇到比较大的grad时，方均根增大，learning rate减小，步长减小，遇到小的grad反之，但是也会发现这样的做法是有一定延迟性的。![alt text](..\notes\pictures\optimize_1.png)

> **RMSProp**：$m_j^i=-{{\eta}\over{\sigma ^i_j}}grad^i_j$,$$\sigma ^i_j=\sqrt{\alpha(\sigma ^{i-1}_{j})^2+(1-\alpha)(grad_j^{i})^2}$$

> **Adam**：RMSProp + Momentum 

---

### other tricks

![alt text](..\notes\pictures\optimize_2.png)