from sklearn.datasets import make_circles, make_moons
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src import models
import os
from src.utils import plot_true_sampled, samples_plot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
parser.add_argument("--model", type=str, default="RealNVP")
args = parser.parse_args()
save = args.save
model_type = args.model

assert model_type in ["RealNVP", "NICE"]

if save:
    SAVE_PATH = os.path.join(os.getcwd(), "figures")
else:
    SAVE_PATH = None

# (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
x, y = make_moons(1000, noise=0.1)

x = tf.convert_to_tensor(x, tf.float32)

if model_type == "RealNVP":
    flow = models.RealNVPFlow(output_dim=2,
                              n_couplings=5,
                              hidden_units=(10, 15, 15),
                              mode="even_odd",
                              activation="relu",
                              distr="logistic")
else:
    flow = models.NICEFlow(output_dim=2,
                           n_couplings=5,
                           hidden_units=(10, 15, 15),
                           mode="even_odd",
                           activation="relu",
                           distr="logistic")
o = flow(x)
flow.compile("adam")

x_ = np.linspace(-10, 10, 300)
xx, yy = np.meshgrid(x_, x_)
xy = np.concatenate([np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))], -1)
xy = tf.convert_to_tensor(xy, tf.float32)

# Untrained Samples
prior_samples = flow.distr_prior.sample(350)
posterior_samples, _ = flow.inverse(prior_samples)
plot_true_sampled(x, posterior_samples, SAVE_PATH, f"{model_type} - Untrained | True vs Model Samples")
xyi, _ = flow.inverse(xy)
llxy = flow.distr_prior.log_prob(xy)
samples_plot(
    xy,
    prior_samples,
    xyi,
    posterior_samples,
    llxy,
    save_path=SAVE_PATH,
    name=f"{model_type} | Samples from Un-Trained Model"
)

# Training
flow.fit(x, epochs=1500, batch_size=x.shape[0])
# Trained Samples
posterior_samples_trained, _ = flow.inverse(prior_samples)
xyi_trained, _ = flow.inverse(xy)
llxy_trained = flow.distr_prior.log_prob(xy)
plot_true_sampled(x, posterior_samples_trained, SAVE_PATH, f"{model_type} - Trained | True vs Model Samples")

samples_plot(
    xy,
    prior_samples,
    xyi_trained,
    posterior_samples_trained,
    llxy_trained,
    save_path=SAVE_PATH,
    name=f"{model_type} | Samples from Trained Model"
)

plt.show()
