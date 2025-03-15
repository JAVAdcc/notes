## GAN

### 经典模型

GAN产生的不是某个特定的东西 而是一组分布
也就是喂进去一组data data本身假设遵循Pdata的分布，我们希望产生的输出服从的分布P_G希望是和Pdata越接近越好
这其实就是对data的极大似然估计

> **hint: Maximum Likelihood Estimation = Minimize KL Divergence**
> $\theta ^{*} = arg \max _{\theta}\Sigma log P_G(x^i;\theta)\approx arg \max _{\theta}E(logP_G(x;\theta)) $
> $=arg \max _{\theta}\int P_{data}logP_G(x;\theta) - \int P_{data}logP_{data}(x)$
> $=arg min KL(P_{data}||P_G)$
> **问题归结于$G^* = arg\min Div(P_G, P_{data})$**

解决方法：训练一个Discriminator
训练目标：看到data的图打高分 看到generate的图打低分 相当于一个classifier
表示出来就是：
$$D^* = arg \max_D V(D,G)$$
$$V = E_{data}(logD(y)) + E_G(log(1-D(y)))$$
**发现**，maxV和JSDivergence有关
问题再转化为：
$$G^* = arg \min_G \max_D V(G,D)$$
$$D^* = arg \max_D V(D,G)$$

按照这样的思路，训练过程基本就是：
> 首先一开始生成器的参数是随机的，先训练辨别器。
> 喂给生成器一些随机的低维向量，生成很多图片，训练辨别器
> 然后训练一定step后反过来训练生成器。
> 固定住D，梯度下降法继续解
> 依此循环


这样就是HW6里最开始的例程，训练效果也是比较差。
**问题在于**训练过程中D被训得非常好不一定是一件好事 反而会造成G的梯度消失
[当Pr与Pg的支撑集（support）是高维空间中的低维流形（manifold）时，Pr与Pg重叠部分测度（measure）为0的概率为1](https://zhuanlan.zhihu.com/p/25071913)
大意是认为G生成的分布始终只是高维空间中的低维流形 因此和真实分布重叠的概率其实很小

### WGAN优化

上述问题的出现与loss的选取有关
WGAN中loss的设计基于Wasserstein distance
**Wasserstein distance**：$$
W(P_r, P_g) = \inf_{\gamma \in \Gamma(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma} [\| x - y \|]
$$
这个线性规划问题有对偶形式（虽然我不懂怎么推的）：
$$
W(P_r, P_g) = \sup_{\| D \|_L \leq 1} \left( \mathbb{E}_{x \sim P_r} [D(x)] - \mathbb{E}_{x \sim P_g} [D(x)] \right)
$$
最终的优化问题转化成min这个W函数：
$$
G^* = arg \min_G \max_{\| D \|_L \leq 1} \left( \mathbb{E}_{real} [D(x)] - \mathbb{E}_{generate} [D(G(z))] \right)
$$
$$
D^* = arg \max_D \left( \mathbb{E}_{real} [D(x)] - \mathbb{E}_{generate} [D(G(z))] \right)
$$
结合代码看一下这个优化过程：

```python
loss_D = - (torch.mean(r_logit) - torch.mean(f_logit))
% min(-)也即max() 固定G下训练D获取W函数
loss_G = -torch.mean(self.D(f_imgs))
% 固定D，这时候只有第二项能动 min第二项 最小化W
```

至此WGAN的代码实现应该是比较明白了，修改的点其实很少：
- 原本D输出的0-1的打分应更换为原始的输出，也就是去掉最后一层sigmoid
- loss中也不再用log，直接改为输出的均值
- 另外是为了满足Lipschitz条件，WGAN的做法是把D的权重全部限制在01之间，认为这样可以限制D(x)的输出值

不过这几个点理解清楚还是比较费工夫的，至少到现在因为数学问题我还不太理解W距离对偶问题是怎么转化的，另外解W距离本身也需要线性规划的方法去做

而WGAN由于Lipschitz条件处理比较粗糙，还有WGAN-gp进一步解决了这一点

