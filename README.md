# Neural Statistician

## About
This repo contains a PyTorch implementation of the model outlined in [Towards a Neural Statistician (Edwards & Storkey, ICLR 2017)](https://arxiv.org/abs/1606.02185). All models are close approximations of those listed in the paper. Where improved performance, reliability etc. were easily available using batch norm, or an extra layer in a particular submodule, the model was adapted accordingly. Differences should be minor and relatively clear.

## Dependencies
The project was written in Python 3.6. You should also use Python 3.6. Installing the latest versions of the following should satisfy dependencies:
  
- pytorch (and a GPU)
- numpy
- scipy
- matplotlib
- scikit-image
- tqdm

Conda is the recommended package manager.

## Structure
The four main experiments are each located in their own subdirectory. Each subdirectory contains the following five files:

- ```data.py``` Methods for creating datasets, and custom providers leveraging PyTorch's ```Dataset``` class. 
- ```nets.py``` Sub-networks used in Statistician model, implemented as subclasses of ```torch.nn.Module```. These correspond to the shared encoder, statistic network, inference network, latent decoder, and observation decoder referred to in the paper.
- ```model.py``` Implementation of the Neural Statistician model.
- ```plot.py``` Utility functions for plotting results.
- ```run.py``` Driver program with tweakable command line args. 

The ```spatial```, ```omniglot```, and ```faces``` directories also contain an additional ```create.py``` file used for data pre-processing. Details are provided below.

## Data
Before running the experiments, you need data. In particular, get the 28x28 greyscale Omniglot from [here](https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT), a gzipped MNIST from [here](http://yann.lecun.com/exdb/mnist/) (or wherever you have it lying around), and the aligned YouTube Faces from [here](https://www.cs.tau.ac.il/~wolf/ytfaces/). As mentioned, there are pre-processing scripts included in the appropriate directories. You'll need to specify directories for each dataset. This is explained in detail below.
 
## Usage
CPU only training is not supported. 
### Synthetic
```python synthrun.py```

![](./synthetic/output/figures/hard-contexts.pdf)

![](./synthetic/output/figures/hard-means.pdf)

![](./synthetic/output/figures/hard-variances.pdf)

### Spatial MNIST
```python spatialrun.py```

### Omniglot
```python omnirun.py```

### YouTube Faces
```python facesrun.py```

## Contact
Please use the [issues tracker](https://github.com/conormdurkan/neural-statistician/issues) for questions, problems etc.