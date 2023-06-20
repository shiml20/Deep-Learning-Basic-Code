Project
- dataset
- output
- plot
- main.py
- network.py
- engine.py
- losses.py
- optimizer.py
- plot.py
- dataset.py
- cmd.sh

I design my project structure as file trees showed above. Each folder or module is illustrated 
below.

- dataset
There are MNIST dataset and CIFAR10 dataset in this folder. Supply the data in these
experiments.

- output
There is some log information outputted by each experiment such as model information, 
hyerparameters and so on.

- plot
This folder saves images plotting loss and accuracy curve in these experiments.

- main.py 
In main.py, we define the model, criterion, optimizer, training process, plot function and some 
log information.

- network.py
In network.py, we define three models needed in these experiments, one MLP model and two 
ConvNets model.

- engine.py
In engine.py, we define train_one_epoch function and training process, in the meanwhile, we 
define validate and test function.

- losses.py
In losses.py, we define different criterion functions include Cross Entropy loss and Euclidean 
loss.

- optimizer.py
In optimizer.py, we define different optimizer include SGD without momentum, SGD with 
momentum, Adam, AdamW and other functions.

- plot.py
In plot.py, we define plot functions to plot loss and accuracy curve in our experiments. This 
module is the same as ploy.py supplied in hw3.

- dataset.py
In dataset.py, we define functions that download the dataset from web automatically and 
construct dataloaders for training and testing.

- cmd.sh
The instructions used in these experiments.
