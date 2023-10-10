import sys
import os
import random

assert len(sys.argv) >= 4, f"usage: python script_train.py task step gpus <other commands>\n" \
                           f"example python script_train.py 15-5 0,1 0,1"
flag = True
DATA_ROOT='/opt/data/private/zzk/dataset/VOCdevkit/VOC2012/'

DATASET="voc"
TASK=sys.argv[1]
EPOCH=50
BATCH=8
LOSS="bce_loss"
LR=0.0001
CROP_SIZE = 513
THRESH=0.7
MEMORY=0
STEPS=sys.argv[2]
GPUS=sys.argv[3]
other_command = ""
CKPT_PATH='checkpoints'
LOG_PATH='checkpoints/logs'
for i in range(4,len(sys.argv)):
    other_command += sys.argv[i]
    other_command += ' '
NAME = ''
for i in range(len(sys.argv)):
    if sys.argv[i] == '--name':
        NAME=sys.argv[i+1]
        break

os.chdir('../')
print(os.getcwd())
print('dataset root:', DATA_ROOT)
if not os.path.exists(CKPT_PATH):
    os.mkdir(CKPT_PATH)
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)


os.system(f"CUDA_VISIBLE_DEVICES={GPUS} python -u main.py --data_root {DATA_ROOT} --model deeplabv3_swin_transformer \
          --gpu_id {GPUS} --step {STEPS} --crop_val --lr {LR} --crop_size {CROP_SIZE}\
    --batch_size {BATCH} --train_epoch {EPOCH}  --loss_type {LOSS} \
    --dataset {DATASET} --task {TASK} "
          f"--overlap --loss_tred "
          f"--lr_policy poly \
    --pseudo --pseudo_thresh {THRESH} "
          f"--freeze_low  "
          f"--bn_freeze  \
    --unknown --w_transfer --amp "
          f"--mem_size {MEMORY} \
    {other_command} "
          f"| tee {LOG_PATH}/{NAME}_{TASK}.txt"

          )




