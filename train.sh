BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.00001
DECAY_STEP=10

#SESSION=1
#EPOCH=43
#CHECKPOINT=1456
#--r True --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \


GPU_ID=4,5,6,7

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset bdd --net hp --hpstage AF3 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda  --mGPUs

