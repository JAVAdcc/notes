## Classification

表征类别的方法：HW1中的hot vector：[0,0,1]

> 与regression的不同：输出的y是一个矢量
> - **softmax函数**：$y_i^{'}={{e^{y_i}}\over{\Sigma e^{y_j}}}\in (0,1)$，简单理解就是把输出y归一化到01之间
> - loss function的改变：**Cross-entropy**：$e=-\Sigma \hat{y_i}lny_i^{'}$

**疑惑**：$\hat{y}$这个矢量中只有一个维度不为0，用cross-entropy算出来的梯度岂不是只有一个维度的方向