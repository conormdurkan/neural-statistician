from matplotlib import pyplot as plt


def grid(inputs, reconstructions, ncols=8):
    fig, axs = plt.subplots(nrows=2, ncols=8)
    inputs = inputs.data.cpu().numpy()
    reconstructions = reconstructions.data.cpu().numpy()

    def plot_single(ax, points):
        ax.scatter(points[:, 0], points[:, 1], s=2, color='black')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, 27])
        ax.set_ylim([0, 27])
        ax.set_aspect('equal', adjustable='box')

    for i in range(ncols):
        # fill one column of subplots per loop iteration
        plot_single(axs[0, i], inputs[i])
        plot_single(axs[1, i], reconstructions[i])

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()