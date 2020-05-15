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
