# EXP1
CUDA_VISIBLE_DEVICES=9 python Deep-Learning-Basic-Code/CV/assemble/imgs_processed \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 \


# CAM
python cam.py



