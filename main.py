from sklearn.datasets import make_circles, make_moons
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src import models
import os
from src.utils import plot_true_sampled, samples_plot, make_spiral_galaxy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
parser.add_argument("--show", action="store_true")
parser.add_argument("--model", type=str, default="RealNVP")
parser.add_argument("--dataset", type=str, default="moons")
parser.add_argument("--layers", type=int, default=5)
args = parser.parse_args()
save = args.save
model_type = args.model
dataset = args.dataset
n_couplings = args.layers
show = args.show

assert model_type in ["RealNVP", "NICE"]
assert dataset in ["moons", "spirals", "circles"]

gpu_device = tf.config.get_visible_devices("GPU")
if gpu_device is not None:
    try:
        tf.config.set_logical_device_configuration(gpu_device[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    except Exception as e:
        raise e


if save:
    SAVE_PATH = os.path.join(os.getcwd(), "figures")
else:
    SAVE_PATH = None

SAVE_PATH = os.path.join(SAVE_PATH, dataset)
if not os.path.exists(SAVE_PATH):
    print(f"Creating path: {SAVE_PATH}")
    os.makedirs(SAVE_PATH)

# (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
if dataset == "moons":
    x, y = make_moons(1000, noise=0.1)
elif dataset == "circles":
    x, y = make_circles(1000, noise=0.1, factor=0.3)
elif dataset == "spirals":
    x, y = make_spiral_galaxy(n_samples=1000, noise=0.1)
else:
    raise ValueError("No dataset")

x = tf.convert_to_tensor(x, tf.float32)

if model_type == "RealNVP":
    flow = models.RealNVPFlow(
        output_dim=2,
        n_couplings=n_couplings,
        hidden_units=(10, 15, 15),
        mode="even_odd",
        activation="relu",
        distr="logistic",
    )
else:
    flow = models.NICEFlow(
        output_dim=2,
        n_couplings=n_couplings,
        hidden_units=(10, 15, 15),
        mode="even_odd",
        activation="relu",
        distr="logistic",
    )
o = flow(x)
flow.compile("adam")
x_ = np.linspace(-10, 10, 300)
xx, yy = np.meshgrid(x_, x_)
xy = np.concatenate([np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))], -1)
xy = tf.convert_to_tensor(xy, tf.float32)

# Untrained Samples
prior_samples = flow.distr_prior.sample(500)
posterior_samples, _ = flow.inverse(prior_samples)
plot_true_sampled(
    x, posterior_samples, save_path=SAVE_PATH, name=f"{model_type} (Untrained) - True vs Model Samples"
)
xyi, _ = flow.inverse(xy)
llxy = flow.distr_prior.log_prob(xy)
samples_plot(
    xy,
    prior_samples,
    xyi,
    posterior_samples,
    llxy,
    save_path=SAVE_PATH,
    name=f"{model_type} - Samples from Un-Trained Model",
)

# Training
flow.fit(x, epochs=2500, batch_size=x.shape[0])
# Trained Samples
posterior_samples_trained, _ = flow.inverse(prior_samples)
xyi_trained, _ = flow.inverse(xy)
llxy_trained = flow.distr_prior.log_prob(xy)
plot_true_sampled(
    x,
    posterior_samples_trained,
    save_path=SAVE_PATH,
    name=f"{model_type} (Trained) - True vs Model Samples",
)

samples_plot(
    xy,
    prior_samples,
    xyi_trained,
    posterior_samples_trained,
    llxy_trained,
    save_path=SAVE_PATH,
    name=f"{model_type} - Samples from Trained Model",
)

if show:
    plt.show()
plt.close()
