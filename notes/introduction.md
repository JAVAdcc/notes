# Introduction

目的:找一个函数
这个函数就是神经网络

输入:vector,matrix,sequence...
输出:数值(回归),类别(分类),text,image...

supervised:
Lecture1-5 supervised learning
Lecture7 self-supervised learning(pre-train)
unsupervised:
Lecture6 generative adversarial network
Lecture12 reinforcement learning

other:
Lecture8 Anomaly Detection
Lecture9 Explainable AI
Lecture10 Model Attack
Lecture11 Domain Adaptation(就是关系那个domain)
Lecture13 Network Compression
Lecture14 Life-long Learning??
Lecture15 Meta learing

---

刚刚看完[deeplearning的简介](https://www.youtube.com/watch?v=bHcJCp2Fyxs&t=4s)
今天是来不及整理了，不过也是有所理解了。
需要明确机器学习的目的：

>找一个复杂的函数

我们最终目的也就是如此，有了这个黑盒子，我给一个输入，这个输入**可能是很多feature（多元函数）**，但是这个盒子一定能给出我想要的结果。
而所谓的激活函数，实际上是对这个目标函数拟合的**组件**，不断调整各个组件（也就是神经元）的参数，逼近我的目标

呃，不过有一点不理解，按照我现在这样的方式理解不了deep的道理，不太清楚多层layer的意义。

> 补充：deep的优势体现在效率上，同等任务下，多层网络的参数会少于单一一层很庞大的网络，例如用逻辑门搭一个判断00100中有多少个0的电路，两种搭建方法使用的门的数量就是指数级和线性级的区别

---

### 模型优化的思路

![alt text](pictures\image.png)
测试集出问题时需要检查训练集，如果误差较大问题就出在模型（增大弹性）或者优化器上。

如果训练集误差小，但是检测效果还是不好，可能是overfitting的问题（极端的情况下就是一个完全没用，只会拟合trainingdata的函数）

>最简单的解决方法：增加training data
>data augmentation：位移，左右翻转，旋转，裁剪
>限制model的弹性：constrain or flexible？
>![alt text](pictures\image1.png)
