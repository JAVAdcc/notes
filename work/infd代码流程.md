��дһ��dataset�Ľ�������

ʹ��ѵ��vae������

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 run.py --cfg cfgs/ae_custom.yaml
```

������ԣ�dataset����wrapper_cae��װ��image_folder
����������̣�

``` python
-> run.py
trainer.run()
-> infd_trainer.py
infd_trainer.run() # ʵ�ʵ��õ���base_trainer.run()
-> base_trainer.py
base_trainer.make_datasets()
for split, spec in cfg.datasets.items():
    # split:{"train", "train_hrft", "val"}
    # spec:{"name", "args":{} , "loader":{}}
    # ����spec['name']Ϊwrapper_cae
    dataset = datasets.make(spec)
    # ��һ���൱��
    # dataset = wraaper_cae(*args)
-> wrapper_cae.py
BaseWrapperCAE.__init__(*args Ҳ����yaml��args�µ�һ�Ѳ���)
self.dataset = datasets.make(dataset)
# �˴�datasetΪargs.dataset = {"name" = "image_folder", "args" = {"root_path", "resize}}
# �൱��
# self.dataset = image_folder(root_path, resize)
->image_folder.py
ImageFolder.__init__(self, root_path, square_crop=True, resize=None, rand_crop=None):
ImageFolder�̳�Dataset�࣬���ǱȽ���Ϥ��Dataset�ˣ�ά����__len__��__getitem__����
__getitem__����һЩ������ǿ�Ĺ���(resize&crop)
```
