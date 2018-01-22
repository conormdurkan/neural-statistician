# Neural Statistician

## About
This repo contains a PyTorch implementation of the model outlined in [Towards a Neural Statistician (Edwards & Storkey, ICLR 2017)](https://arxiv.org/abs/1606.02185). All models are close approximations of those listed in the paper.

The code was written using Python 3.6, and only works with a GPU setup as is.

## Structure
The four main experiments are each located in their own subdirectory. Each subdirectory contains the following five files:

- ```data.py``` Methods for creating datasets, and custom providers leveraging PyTorch's ```Dataset``` class. 
- ```nets.py``` Subnetworks used in Statistician model, implemented as subclasses of ```torch.nn.Module```. These correspond to the shared encoder, statistic network, inference network, latent decoder, and observation decoder referred to in the paper.
- ```model.py``` Implementation of the Neural Statistician model.
- ```plot.py``` Utility functions for plotting results.
- ```run.py``` Driver program with tweakable command line args. 

The ```spatial```, ```omniglot```, and ```faces``` directories also contain an additional ```create.py``` file used for data preprocessing.

## Data
The following data sets are used:

- 28 x 28 greyscale Omniglot from [here](https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT)
- MNIST from [here](http://yann.lecun.com/exdb/mnist/)
- aligned YouTube Faces from [here](https://www.cs.tau.ac.il/~wolf/ytfaces/)

Further preprocessing is needed. As mentioned, there are preprocessing scripts included in the appropriate directories.

## Contact
Please use the [issues tracker](https://github.com/conormdurkan/neural-statistician/issues) for questions, problems etc.