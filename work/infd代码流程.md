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
```
