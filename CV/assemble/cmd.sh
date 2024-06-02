# EXP1
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 \
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd_0_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 0 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout01_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.1 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout02_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.2 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optadaw \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt adamw \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr01_wd0_dropout00_optadaw \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt adamw \
    --dropout 0.0 -wd 0 ;\

# EXP2
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr005_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 256 \
    --epochs 15 -lr 0.05 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs15_lr001_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 256 \
    --epochs 15 -lr 0.01 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs30_lr005_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 256 \
    --epochs 30 -lr 0.05 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\

# EXP3
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs30_lr001_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 30 -lr 0.01 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\

CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet34_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet34 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet18_oridata_epochs50_lr01_wd0_dropout00_optadaw \
    --arch resnet18 --batch_size 384 \
    --epochs 50 -lr 0.1 --opt adamw \
    --dropout 0.0 -wd 0 ;\
CUDA_VISIBLE_DEVICES=8 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet34_oridata_epochs30_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet34 --batch_size 384 \
    --epochs 30 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=8 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/resnet34_oridata_epochs15_lr005_wd_1e-4_dropout00_optsgd \
    --arch resnet34 --batch_size 384 \
    --epochs 15 -lr 0.05 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\


# EXP4
CUDA_VISIBLE_DEVICES=8 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/sml2_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch sml --batch_size 64 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=8 python main_aa.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/aa3_resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=8 python main_aa.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/aa3_resnet34_oridata_epochs30_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet34 --batch_size 384 \
    --epochs 30 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main_aa2.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/aa2_resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\
CUDA_VISIBLE_DEVICES=9 python main_aa2.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/aa2_resnet34_oridata_epochs30_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet34 --batch_size 384 \
    --epochs 30 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\

CUDA_VISIBLE_DEVICES=8 python main_aa.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/aa3_resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd_v2 \
    --resume logs/aa3_resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd/checkpoint-best.pth \
    --arch resnet18 --batch_size 384 \
    --epochs 30 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\

# EXP 5
CUDA_VISIBLE_DEVICES=9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/473_resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch resnet18 --batch_size 384 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 \
    --num_classes 473 ;\

# EXP6
CUDA_VISIBLE_DEVICES=8,9 python main.py /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --output_dir logs/sml2_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd \
    --arch sml --batch_size 64 \
    --epochs 15 -lr 0.1 --opt sgd \
    --dropout 0.0 -wd 1e-4 ;\

# CAM
CUDA_VISIBLE_DEVICES=9 python main.py -e /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --resume logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optsgd/checkpoint-best.pth --opt sgd
CUDA_VISIBLE_DEVICES=9 python main.py -e /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --resume logs/resnet18_oridata_epochs15_lr01_wd_0_dropout00_optsgd/checkpoint-best.pth --opt sgd
CUDA_VISIBLE_DEVICES=9 python main.py -e /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --resume logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout01_optsgd/checkpoint-best.pth --opt sgd
CUDA_VISIBLE_DEVICES=9 python main.py -e /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --resume logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout02_optsgd/checkpoint-best.pth --opt sgd
CUDA_VISIBLE_DEVICES=9 python main.py -e /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --resume logs/resnet18_oridata_epochs15_lr01_wd_1e-4_dropout00_optadaw/checkpoint-best.pth --opt adamw
CUDA_VISIBLE_DEVICES=9 python main.py -e /cluster/home1/wyl/spatial_denamic_v1/Homework_sml/data/ \
    --resume logs/resnet18_oridata_epochs15_lr01_wd0_dropout00_optadaw/checkpoint-best.pth --opt adamw





