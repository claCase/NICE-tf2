from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os


def samples_plot(
        priorxy, prior_samples, posteriorxy, posterior_samples, llxy, save_path=None, name="NICE"
):
    fig, ax = plt.subplots(2, figsize=(10, 10))
    if name is not None:
        fig.suptitle(name)
    ax[0].set_title("Samples From Prior (hidden space)")
    img0 = ax[0].scatter(priorxy[:, 0], priorxy[:, 1], c=llxy, cmap="inferno")
    ax[0].scatter(prior_samples[:, 0], prior_samples[:, 1], color="black", s=3)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax0, orientation="vertical")

    ax[1].set_title("Samples From Posterior (data space)")
    img1 = ax[1].scatter(posteriorxy[:, 0], posteriorxy[:, 1], c=llxy, cmap="inferno")
    ax[1].scatter(
        posterior_samples[:, 0], posterior_samples[:, 1], color="black", s=3
    )
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax1, orientation="vertical")
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, name + ".png"))


def plot_true_sampled(true, samples, save_path=None, name="True vs Model Samples"):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(name)
    plt.scatter(true[:, 0], true[:, 1], label="True Data", color="green")
    plt.scatter(samples[:, 0], samples[:, 1], label="Model Samples", color="red")
    plt.legend(loc="best")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, name + ".png"))
