from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle as skshuffle


def samples_plot(
        priorxy,
        prior_samples,
        posteriorxy,
        posterior_samples,
        llxy,
        save_path=None,
        name="NICE",
):
    if isinstance(priorxy, tf.Tensor):
        priorxy = priorxy.numpy()
    if isinstance(llxy, tf.Tensor):
        llxy = llxy.numpy()
    xx_shape = int(np.sqrt(priorxy.shape[0]))
    priorxy = np.split(priorxy.reshape(xx_shape, xx_shape, 2), 2, axis=-1)
    priorxx, prioryy = priorxy[0].squeeze(), priorxy[1].squeeze()
    priorzz = llxy.reshape(xx_shape, xx_shape)

    fig, ax = plt.subplots(2, figsize=(10, 10))
    if name is not None:
        fig.suptitle(name)
    ax[0].set_title("Samples From Prior (hidden space)")
    img0 = ax[0].contourf(priorxx, prioryy, priorzz, cmap="inferno")
    ax[0].scatter(prior_samples[:, 0], prior_samples[:, 1], color="black", s=3)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax0, orientation="vertical")

    if isinstance(posteriorxy, tf.Tensor):
        posteriorxy = posteriorxy.numpy()
    xx_shape = int(np.sqrt(posteriorxy.shape[0]))
    posteriorxy = np.split(posteriorxy.reshape(xx_shape, xx_shape, 2), 2, axis=-1)
    posteriorxx, posterioryy = posteriorxy[0].squeeze(), posteriorxy[1].squeeze()

    ax[1].set_title("Samples From Posterior (data space)")
    img1 = ax[1].contourf(posteriorxx, posterioryy, priorzz, cmap="inferno")
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


def make_spiral_galaxy(n_spirals=5, length=1, angle=np.pi / 2, n_samples=100, noise=0, shuffle=True):
    thetas = np.linspace(0, np.pi * 2 * (n_spirals - 1) / n_spirals, n_spirals)
    radius = np.linspace(np.zeros(len(thetas)) + 0.1, np.ones(len(thetas)) * length + 0.1, n_samples)
    angles = np.linspace(thetas, thetas + angle, n_samples)
    if noise:
        angles += np.random.normal(size=angles.shape) * noise * np.linspace(1.5, .1, n_samples)[:, None]
    x0 = np.cos(angles) * radius
    x1 = np.sin(angles) * radius
    x0 = x0.T.reshape(-1, 1)
    x1 = x1.T.reshape(-1, 1)
    xy = np.concatenate([x0, x1], -1)
    y = np.repeat(np.arange(n_spirals), n_samples)
    if shuffle:
        xy, y = skshuffle(xy, y)
    return xy, y
