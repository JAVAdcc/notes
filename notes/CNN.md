## Convolutional neural network

---

### 卷积层

卷积神经网络首先一个非常显著的特征是网络结构的改变。

> **Convolutional layer**：关注与Linear的区别
> - **Receptive Field**：卷积层的neuron并不是和所有输入的feature相连，而仅仅与一个3×3的Receptive Field相连。这个东西在整张图中的分布并非完全不相交，而是通过步长stride（通常为1，2）平移遍历整张图
> - **Parameter sharing**：控制同一片receptive field的neuron的参数一致
> 
> 卷积层中的neuron被称作filter，感觉也确实，操作起来就是很多个滤波器划过整张图，那同一个滤波器的参数肯定是一致的

> **Insight**：两个动机，第一是我们不用看整张图片，第二是同一个pattern可能出现在不同的区域，这两个动机形成了卷积层的两个特点
> 另外说明，每个filter扫过图片的过程就是卷积

>一个例子：比如一张黑白图片，表示出来是一个6×6的矩阵，我用一个3×3×1的filter（也算是kernel吧）（注意这个1，是因为原图片只有一个channel），按stride = 1扫一遍出来一个4×4的矩阵，总共64个filter，那我的结果就是一张新的，有64个channel的图片 。

---

### 池化

目的很朴素：把通过卷积层产生的图片变小，压缩，subsampling。
方法也很朴素，比如max pooling就是分割一下图片，在分割区域内的几个输出的feature中挑一个最大值作为代表

---

全过程就是![alt text](..\notes\pictures\CNN_1.png)

---

存在的问题：不能处理旋转放大的图片

解决方式：添加spatial transformer
作用：裁剪&旋转
对于一张图，想要实现这个操作只需要对每个像素点的位置(x,y)施以变换
$$\left[\begin{matrix}
    a, b \\
    c, d
\end{matrix}
    \right] \cdot 
\left[\begin{matrix}
    x\\
    y
\end{matrix}
    \right] + 
\left[\begin{matrix}
    e\\
    f
\end{matrix}
    \right]$$
spatial transformer一层的神经网络就是要确定出这个变换的6个参数