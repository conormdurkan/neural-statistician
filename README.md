# Neural Statistician

## About
This repo contains a PyTorch implementation of the model outlined in [Towards a Neural Statistician](https://arxiv.org/abs/1606.02185). All models are close approximations of those listed in the paper. Where improved performance was easily available using methods like Batch Normalization, the models were augmented accordingly. 

## Structure
The four main experiments are each located in their own subdirectory. Each subdirectory contains the following five files:

- ```data.py``` Methods for creating datasets, and custom providers leveraging PyTorch's ```Dataset``` class. 
- ```nets.py``` Sub-networks used in Statistician model, implemented as subclasses of ```torch.nn.Module```. These correspond to the shared encoder, statistic network, inference network, latent decoder, and observation decoder referred to in the paper.
- ```model.py``` Implementation of the Neural Statistician model.
- ```plot.py``` Utility functions for plotting results.
- ```run.py``` Driver program with tweakable command line args. 
 
## Dependencies
This was written in Python 3.6, but presumably runs with most other versions. Installing the latest version of the following should satisfy dependencies.
  
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Data
Before running the experiments, you need data. In particular, get the 28x28 greyscale Omniglot from [here](https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT), a gzipped MNIST from wherever you have it lying around, and the aligned YouTube Faces from [here](https://www.cs.tau.ac.il/~wolf/ytfaces/). There are pre-processing scripts included in the appropriate directories. You'll need to specify the correct input and output locations for each dataset. 
 
## Experiments
### Synthetic
```python synthrun.py```

### Spatial MNIST
```python spatialrun.py```

### Omniglot
```python omnirun.py```

### YouTube Faces
```python facesrun.py```