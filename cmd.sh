# CNN1
python main.py --lr 0.001 --epoch 20 --opt sgd --dataset mnist ;\
python main.py --lr 0.003 --epoch 20 --opt sgd --dataset mnist ;\
python main.py --lr 0.005 --epoch 20  --opt sgd --dataset mnist ;\
python main.py --lr 0.01 --epoch 20  --opt sgd --dataset mnist ;\
python main.py --lr 0.001 --epoch 20  --opt adam --dataset mnist ;\
python main.py --lr 0.003 --epoch 20  --opt adam --dataset mnist ;\
python main.py --lr 0.005 --epoch 20  --opt adam --dataset mnist ;\
python main.py --lr 0.01 --epoch 20  --opt adam --dataset mnist ;\

python main.py --lr 0.001 --epoch 20  --opt sgd --dataset cifar10 ;\
python main.py --lr 0.003 --epoch 20  --opt sgd --dataset cifar10 ;\
python main.py --lr 0.005 --epoch 20  --opt sgd --dataset cifar10 ;\
python main.py --lr 0.01 --epoch 20  --opt sgd --dataset cifar10 ;\
python main.py --lr 0.001 --epoch 20  --opt adam --dataset cifar10 ;\
python main.py --lr 0.003 --epoch 20  --opt adam --dataset cifar10 --batch_size 128;\
python main.py --lr 0.005 --epoch 20  --opt adam --dataset cifar10 --batch_size 128;\
python main.py --lr 0.01 --epoch 20  --opt adam --dataset cifar10 --batch_size 128;\

python main.py --lr 0.003 --epoch 50  --opt adam --dataset cifar10 --model cnn2 --batch_size 128;\
python main.py --lr 0.005 --epoch 50  --opt adam --dataset cifar10 --model cnn2 --batch_size 128;\

# CNN2
python main.py --lr 0.001 --epoch 20 --opt adam --dataset mnist --model cnn2;\
python main.py --lr 0.003 --epoch 20 --opt adam --dataset mnist --model cnn2;\
python main.py --lr 0.005 --epoch 20 --opt adam --dataset mnist --model cnn2;\
python main.py --lr 0.007 --epoch 20 --opt adam --dataset mnist --model cnn2;\
python main.py --lr 0.01 --epoch 20 --opt adam --dataset mnist --model cnn2;\

python main.py --lr 0.001 --epoch 20 --opt adam --dataset cifar10 --model cnn2;\
python main.py --lr 0.003 --epoch 20 --opt adam --dataset cifar10 --model cnn2;\
python main.py --lr 0.005 --epoch 20 --opt adam --dataset cifar10 --model cnn2;\
python main.py --lr 0.007 --epoch 20 --opt adam --dataset cifar10 --model cnn2;\
python main.py --lr 0.01 --epoch 20 --opt adam --dataset cifar10 --model cnn2;\

python main.py --lr 0.0005 --epoch 50 --opt adam --dataset cifar10 --model cnn2 ;\
python main.py --lr 0.001 --epoch 50 --opt adam --dataset cifar10 --model cnn2;\
python main.py --lr 0.003 --epoch 50 --opt adam --dataset cifar10 --model cnn2;\
python main.py --lr 0.005 --epoch 50 --opt adam --dataset cifar10 --model cnn2;
python main.py --lr 0.001 --epoch 50 --opt adam --dataset cifar10 --model cnn2 --weight_decay 0;\
python main.py --lr 0.003 --epoch 50 --opt adam --dataset cifar10 --model cnn2 --weight_decay 0;\


# MLP1
python main.py --lr 0.001 --epoch 15 --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.003 --epoch 15 --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.005 --epoch 15  --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.01 --epoch 15  --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.001 --epoch 15  --opt adam --dataset mnist --model mlp1;\
python main.py --lr 0.003 --epoch 15  --opt adam --dataset mnist --model mlp1;\
python main.py --lr 0.005 --epoch 15  --opt adam --dataset mnist --model mlp1;\
python main.py --lr 0.01 --epoch 15  --opt adam --dataset mnist --model mlp1;\

python main.py --lr 0.001 --epoch 20 --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.003 --epoch 20 --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.005 --epoch 20  --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.01 --epoch 20  --opt sgd --dataset mnist --model mlp1;\
python main.py --lr 0.001 --epoch 20  --opt adam --dataset mnist --model mlp1;\
python main.py --lr 0.003 --epoch 20  --opt adam --dataset mnist --model mlp1;\
python main.py --lr 0.005 --epoch 20  --opt adam --dataset mnist --model mlp1;\
python main.py --lr 0.01 --epoch 20  --opt adam --dataset mnist --model mlp1;\


python main.py --lr 0.001 --epoch 15 --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.003 --epoch 15 --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.005 --epoch 15  --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.01 --epoch 15  --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.001 --epoch 15  --opt adam --dataset cifar10 --model mlp1;\
python main.py --lr 0.003 --epoch 15  --opt adam --dataset cifar10 --model mlp1;\
python main.py --lr 0.005 --epoch 15  --opt adam --dataset cifar10 --model mlp1;\
python main.py --lr 0.01 --epoch 15  --opt adam --dataset cifar10 --model mlp1;\

python main.py --lr 0.001 --epoch 20 --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.003 --epoch 20 --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.005 --epoch 20  --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.01 --epoch 20  --opt sgd --dataset cifar10 --model mlp1;\
python main.py --lr 0.001 --epoch 20  --opt adam --dataset cifar10 --model mlp1;\
python main.py --lr 0.003 --epoch 20  --opt adam --dataset cifar10 --model mlp1;\
python main.py --lr 0.005 --epoch 20  --opt adam --dataset cifar10 --model mlp1;\
python main.py --lr 0.01 --epoch 20  --opt adam --dataset cifar10 --model mlp1;\