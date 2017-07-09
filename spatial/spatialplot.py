from matplotlib import pyplot as plt


def grid(inputs, reconstructions, ncols=8):
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols, 2))
    inputs = inputs.data.cpu().numpy()
    reconstructions = reconstructions.data.cpu().numpy()

    def plot_single(ax, points):
        ax.scatter(points[:, 0], points[:, 1], s=5,  color='C0')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, 27])
        ax.set_ylim([0, 27])
        ax.set_aspect('equal', adjustable='box')

    for i in range(ncols):
        # fill one column of subplots per loop iteration
        plot_single(axs[0, i], inputs[i])
        plot_single(axs[1, i], reconstructions[i])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()