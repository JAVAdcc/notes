## VAE

所谓生成式的模型，可以认为是输入和输出都是一个概率分布
比如说我输入是狗的分布的采样 那我希望输出的图像也落在这个狗的分布中，这样就可以生成很多狗的图片
从输入图片的角度讲，在这么高维的空间中根本不知道狗的图像是怎样的分布，auto encoder感觉呈现了最朴素的一种想法，把高维映射到低维的潜在信息，也就是z，最后把z映射回图像，如果能够明确在z的空间对应不同图像的分布（后验分布），或许就可以完成生成的目标
但这个做法的问题在于，z这个低维空间中的分布也并不美妙，一个图像转化过去的z稍作改变最终就会映射到截然不同的东西上
vae做的事情就是organize latent space，假定z的实际分布就是一个标准正态分布，而encoder也不再是生成一个指定的z，而是生成一个z的正态分布（实现时是生成了这个分布的两个参数），采样出具体的z，再映射回图像空间
loss function由两部分组成，传统的一部分是输出和输入的L2，另一部分是生成的z分布和标准正态分布的KL散度，用来对生成的z分布做一个regularization
这个loss的来源是所谓的ELBO evidence的最低下界，evidence应该就是目标的优化函数，即$log p_\theta (x)$ 在获得后验分布后重建出原始x的概率，ELBO是这个函数的最低下界，我们只需要优化这个ELBO
![alt text](.\pictures\vae_p1.png)
![alt text](.\pictures\vae_p2.png)
![alt text](.\pictures\vae_p3.png)

### VQ_VAE

![alt text](.\pictures\vae_p4.jpg)
~~感觉和VAE好像挺不一样的 不知道是不是理解不太到位~~
有一个地方写的不太对 输入的WH和转成latent code的WH还是不一样的 encoder里会有conv层  latent code本质上已经是提取出的图片的features

想了一下，针对划掉的那一行需要说明一下
这种感觉上的不一样主要还是由于隐空间结构的改变，简单起见可以考虑各自二维的隐空间
VAE就很直观，理想中隐空间服从一个二维的标准正态分布，此时latentcode的形式就是单个z向量，隐空间中每一点（也就是每个z向量）都对应一张输出图片，单个z就蕴含了输出图片全局特征的信息
而vqVAE，首先这个二维隐空间中只有很多离散的点，此时latentcode的形式是一个H * W * D的网格，量化之后其实我就选取了很多隐空间中的点，这些点共同输出一张图片。
也就是说，我在codebook中预先设定的点，并不是单一一个就能独立生成图片，而是多个点共同决定输出。更明确一点，隐空间中单个点只包含了输出的局部特征的信息。
从中可以看出vqvae效果上和vae的不同，vae强制规定隐空间为正态分布的形状，基于这一点获得了性质良好的连续的隐空间，但是也因为平滑的特性会输出模糊的图片。
而vqvae的想法是现实中的特征大多是不连续的，就像我要生成狗和猫的图片，我其实不需要介于两者之间的东西，所以用隐空间中离散的点控制输出可以使得输出更加清晰明确。而且我感觉是不是vqvae能将图中的特征和图片解耦呢。
说实话具体怎么生成我还不是很明白

#### VQ_VAE&基于EMA迭代codebook

首先回顾vqVAE中codebook中embedding vector在优化时在做什么：趋向于输入的latent code
这么说其实是很模糊的，更加确切地描述，是每个embedding vector会更加接近离它最近的那个input latent code（当然不一定只有一个），但是我想这样做法也有一个问题，一部分一直没用到的vector实际上永远不会被更新，因为计算mse只会用到与输入有关联的那些向量

而ema（也不知道为什么叫这个）不基于梯度，而是用直接的方式实现embedding vector接近最近的输入向量
方法是统计每个embedding vector对应的输入向量，取他们的均值作为这个向量的更新值，当然还会有decay控制历史向量值得权重，用epsilon控制除零
其实挺直观的，但是感觉看代码体验有点差

```python
# Use EMA to update the embedding vectors
if self.training:
    self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                (1 - self._decay) * torch.sum(encodings, 0)

    # Laplace smoothing of the cluster size
    n = torch.sum(self._ema_cluster_size.data)
    self._ema_cluster_size = (
        (self._ema_cluster_size + self._epsilon)
        / (n + self._num_embeddings * self._epsilon) * n)

    dw = torch.matmul(encodings.t(), flat_input)
    # dw K*D 就是每个嵌入向量对应到的输入向量之和 后续作为新一轮权重
    self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

    self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))  
```
