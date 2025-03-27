先写一下dataset的建立流程

使用训练vae的命令

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 run.py --cfg cfgs/ae_custom.yaml
```

整体而言，dataset是用wrapper_cae包装的image_folder
具体代码流程：

``` python
-> run.py
trainer.run()
-> infd_trainer.py
infd_trainer.run() # 实际调用的是base_trainer.run()
-> base_trainer.py
base_trainer.make_datasets()
for split, spec in cfg.datasets.items():
    # split:{"train", "train_hrft", "val"}
    # spec:{"name", "args":{} , "loader":{}}
    # 其中spec['name']为wrapper_cae
    dataset = datasets.make(spec)
    # 这一步相当于
    # dataset = wraaper_cae(*args)
-> wrapper_cae.py
BaseWrapperCAE.__init__(*args 也就是yaml的args下的一堆参数)
self.dataset = datasets.make(dataset)
# 此处dataset为args.dataset = {"name" = "image_folder", "args" = {"root_path", "resize}}
# 相当于
# self.dataset = image_folder(root_path, resize)
->image_folder.py
ImageFolder.__init__(self, root_path, square_crop=True, resize=None, rand_crop=None):
ImageFolder继承Dataset类，就是比较熟悉的Dataset了，维护了__len__和__getitem__方法
__getitem__会做一些数据增强的工作(resize&crop)

返回阶段
最终trainer的属性中将存储dataset和loader dataset的类型是wrapperCAE
dataset = datasets.make(spec)
self.datasets[split] = dataset
self.loaders[split], self.loader_samplers[split] = self.make_distributed_loader(
    dataset, loader_spec.batch_size, drop_last, shuffle, loader_spec.num_workers)

在从dataset中取data时，将同时调用wrapperCAE和ImageFolder的__getitem__方法
其中wrapperCAE的getitem返回的不单是图片
data = {
    'inp': inp
    'gt': gt_patch, # 3 p p
    'gt_coord': coord, # p p 2
    'gt_cell': cell, # p p 2
}

```
