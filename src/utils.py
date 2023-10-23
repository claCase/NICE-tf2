from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
import tensorflow as tf


def samples_plot(
    priorxy,
    prior_samples,
    posteriorxy,
    posterior_samples,
    llxy,
    save_path=None,
    name="NICE",
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
    ax[1].scatter(posterior_samples[:, 0], posterior_samples[:, 1], color="black", s=3)
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


def split_mode(inputs, mode):
    if mode == "split":
        h1, h2 = tf.split(inputs, 2, axis=-1)
    elif mode == "even_odd":
        h1, h2 = inputs[..., ::2], inputs[..., 1::2]
    else:
        raise ValueError(f"Mode {mode} not supported")
    return h1, h2


def recombine_mode(inputs, mode):
    a, b = inputs
    x, y = tf.shape(a)[0], tf.shape(a)[1]
    if mode == "split":
        return tf.concat([a, b], -1)
    elif mode == "even_odd":
        new_tensor = tf.zeros(shape=(x, y * 2), dtype=a.dtype)
        even = tf.range(y) * 2
        even = tf.tile(even[None, :], (x, 1))
        odd = even + 1
        batch_indices = tf.tile(tf.range(x)[:, None], (1, y))
        even_indices = tf.concat(
            [tf.reshape(batch_indices, (-1, 1)), tf.reshape(even, (-1, 1))], -1
        )
        odd_indices = tf.concat(
            [tf.reshape(batch_indices, (-1, 1)), tf.reshape(odd, (-1, 1))], -1
        )
        new_tensor = tf.tensor_scatter_nd_update(
            new_tensor, even_indices, tf.reshape(a, shape=(-1,))
        )
        new_tensor = tf.tensor_scatter_nd_update(
            new_tensor, odd_indices, tf.reshape(b, shape=(-1,))
        )
        return new_tensor
    else:
        raise ValueError(f"Mode {mode} not supported")
