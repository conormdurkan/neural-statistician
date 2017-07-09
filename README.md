This repo contains a PyTorch implementation of the model outlined in [Towards a Neural Statistician](https://arxiv.org/abs/1606.02185).

The four main experiments are each located in their own subdirectory. Each subdirectory contains the following five files:

- ```data.py``` Methods for creating datasets, and custom providers leveraging PyTorch's ```Dataset``` class. 
- ```nets.py``` Sub-networks used in Statistician model, implemented as subclasses of ```torch.nn.Module```. These correspond to the shared encoder, statistic network, inference network, latent decoder, and observation decoder referred to in the paper.
- ```model.py``` Implementation of the Neural Statistician model.
- ```plot.py``` Utility functions for plotting results.
- ```run.py``` Driver program with tweakable command line args. 

All networks are close approximations of those listed in the paper. Where improved performance was available using methods like Batch Normalization, the model was augmented accordingly. 
 
# Synthetic
```python synthrun.py```

# Spatial MNIST
```python spatialrun.py```

# Omniglot
```python omnirun.py```

# YouTube Faces
```python facesrun.py```