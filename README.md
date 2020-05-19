- Modified from original git repo:https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0. ATTENTION: pytorch 1.0 branch, and prepare the project with pytorch-1.0 README.md

### Preparation
- Prepare data and vgg pretrained model
First of all, clone the code
```shell script
git clone git@github.com:TianmingQiu/pedestrian_detection_attention.git
```
Then, create a folder:
```shell script
cd pedestrian_detection_attention && mkdir data
```
Download dataset and pre-trained model:
```shell script
cd data
```
Download [bdd_data](https://www.dropbox.com/s/yhy179kakprsar1/bdd_data.zip?dl=0) and [pretrained_model](https://www.dropbox.com/s/nf94mno6mu5y1g1/pretrained_model.zip?dl=0) here.
The directroy looks like:
```shell script
data
├── bdd_data
└── pretrained_model
```
### Compilation
```shell script
pip install -r requirements.txt
cd lib
python setup.py build develop
```

If you have the issue with "cannot import .mask", you need to prepare COCO API:
```shell script
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
```

### Train AF3
Check `train.sh` and run `sh train.sh`


- test your training e.g.:
```shell script
python trainval_net.py --dataset pascal_voc --net vgg16 --bs 24 --nw 8 --lr 0.001 --lr_decay_step 1000 --cuda --mGPUs
```


### Custom dataset adaptation:
- Convert BDD annotations to 'xml': https://github.com/Ugenteraan/bdd_json_to_xml/blob/master/convert.py
- BDD dataset class: https://github.com/nishankjain/faster-rcnn-bdd/tree/master/lib/model
- https://github.com/deboc/py-faster-rcnn/tree/master/help
- The first thing you need to do is writing a customized data loader for your own dataset. You can refer to the data loader for pascal_voc and coco in lib/datasets/pascal_voc.py and lib/datasets/coco.py.


### HydraPlus detector:
Train the MNet branch of HydraPlus net:
```shell script
BATCH_SIZE=128
WORKER_NUMBER=4
LEARNING_RATE=0.01
DECAY_STEP=10

GPU_ID=0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset bdd --net hp --hpstage MNet \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda  --mGPUs
```
