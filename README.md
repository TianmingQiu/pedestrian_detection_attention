- Original git repo:https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

- ATTENTION: pytorch 1.0 branch, and prepare the project with pytorch-1.0 README.md

- Prepare data and vgg pretrained model

- test your training e.g.:
```shell script
python trainval_net.py --dataset pascal_voc --net vgg16 --bs 24 --nw 8 --lr 0.001 --lr_decay_step 1000 --cuda --mGPUs
```

- If you have the issue with "cannot import .mask", you need to prepare COCO API:
```shell script
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
```




