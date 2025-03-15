## Cnov2d

![alt text](.\pictures\a_c_p1.png)
这张图已经很好地展示了卷积的过程，不过整个过程的shape变化还是要说明
以Conv2d函数说明
torch里一般期望输入格式(batch_size, channels(input_dim), H, W),两个L就是长和宽
函数作用于第二个维度channels，首先确定的是函数的前两个参数，input_dim，output_dim，这两个参数会把第二个维度转成output_dim，剩余的参数就作用于图片的格式变化，公式如下：
$$H_{out}=⌊{{H_{in}+2P−K}\over{S}}⌋+1$$
操作起来就是先把原始图片周围补上padding圈，再按stride去卷积 
比如[64, 3, 4, 4]->Conv2d(3,4,3,2,1)->[64, 4, 2, 2]

## Separable Convolution 
一个Conv2d实现的操作可以拆成两个Conv2d
分别是pointwise convolution和depthwise convolution

### Pointwise
![alt text](.\pictures\a_c_p2.png)
Conv2d(3, 4, 1, 1, 0)
指定kernelsize为1 逐点算

### Depthwise
![alt text](.\pictures\a_c_p3.png)
Conv2d(input_dim, input_dim, **groups=input_dim**)
重点只在于要指定groups参数，把channel分割成input_dim组，一一操作

## Convtranspose2d

首先记住图片尺寸变化的公式$$output_{size} = (input_{size} - 1)*stride + kernel_{size} - 2*padding + output_{padding}$$
其实意思就是反卷积的过程 先走input-1次stride 最后一次直接补上一个kernel的大小，撑满框框，然后再剪掉padding的圈圈，加上outputpadding
channel的变化还是一样的，也没什么好说的，主要是卷积的过程和stride，padding的操作，具体如下：
![alt text](.\pictures\a_c_p4.jpg)
