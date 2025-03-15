## 讨论batch

我们把一整笔资料拆成很多笔喂给模型，这就是一个一个batch
处理每一个batch时，比如说batch_size是100，我把X和y都叠成100层，那我算loss的时候就比较$y_{pred}和y_{valid}$这两个张量的loss（比如说就算一个MSE）
那我batch_size不一样，这个loss的效果和一个epoch里update的次数都会改变。

> 时间方面的考虑：在batch_size较小时，事实上由于GPU的并行计算，计算一个batch的时间是相近的，所以大的batch_size在时间方面是占优的
> 效果方面的考虑：batch_size越大，optimization越差，loss也会越大

>**Insight**：![alt text](..\notes\pictures\batch_1.png) 
>使用batch时，每次update对应的loss function都是不一样的，所以不容易卡在local minima，对训练有好处。

### conclusion

![alt text](..\notes\pictures\batch_2.png)
因此batch_size是一个hyper_parameter