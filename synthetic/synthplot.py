import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


def scatter_contexts(contexts, labels, distributions, savepath=None):
    """
    
    :param contexts: 
    :param labels: 
    :param distributions: 
    :param savepath: 
    :return: 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    contexts = np.array(contexts).reshape(-1, 3)

    n = len(contexts)
    labels = labels[:n]
    ix = [np.where(labels == label)
          for i, label in enumerate(distributions)]
    colors = [
        'indianred',
        'forestgreen',
        'gold',
        'cornflowerblue',
        'darkviolet'
    ]

    for label, i in enumerate(ix):
        ax.scatter(contexts[i][:, 0], contexts[i][:, 1], contexts[i][:, 2],
                   label=distributions[label].title(),
                   color=colors[label])
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    plt.legend(loc='upper left')

    plt.show()

    if savepath is not None:
        plt.savefig(savepath)


def contexts_by_moment(data, moments, savepath=None, which='mean'):
    """
    
    :param data: 
    :param moments: 
    :param savepath: 
    :param which: 
    :return: 
    """
    # TODO: Fix this
    pass
    # data = ns.test2c()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=provider.data_set['means'],
    #            norm=Normalize)
    #
    # plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
    #                 right='off', left='off', labelleft='off')
    # plt.legend(loc='upper left')
    #
    # plt.show()
    # plt.savefig(savepath)
