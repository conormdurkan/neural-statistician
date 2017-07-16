from matplotlib import pyplot as plt


def grid(inputs, samples, summaries=None, save_path=None, ncols=10):

    inputs = inputs.data.cpu().numpy()
    samples = samples.data.cpu().numpy()
    if summaries is not None:
        summaries = summaries.data.cpu().numpy()
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols, 2))

    def plot_single(ax, points, s, color):
        ax.scatter(points[:, 0], points[:, 1], s=s,  color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, 27])
        ax.set_ylim([0, 27])
        ax.set_aspect('equal', adjustable='box')

    for i in range(ncols):
        # fill one column of subplots per loop iteration
        plot_single(axs[0, i], inputs[i], s=5, color='C0')
        plot_single(axs[1, i], samples[i], s=5, color='C1')
        if summaries is not None:
            plot_single(axs[1, i], summaries[i], s=10, color='C2')

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
