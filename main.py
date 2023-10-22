from sklearn.datasets import make_circles, make_moons
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src import models
import os
from src.utils import plot_true_sampled, samples_plot

SAVE_PATH = os.path.join(os.getcwd(), "figures")

# (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
x, y = make_moons(1000, noise=0.1)

x = tf.convert_to_tensor(x, tf.float32)
nice = models.NICE(
    output_dim=2,
    n_couplings=5,
    hidden_units=(10, 15, 15),
    mode="even_odd",
    activation="relu",
    distr="logistic",
)

o = nice(x)
nice.compile("adam")

x_ = np.linspace(-10, 10, 300)
xx, yy = np.meshgrid(x_, x_)
xy = np.concatenate([np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))], -1)
xy = tf.convert_to_tensor(xy, tf.float32)

# Untrained Samples
prior_samples = nice.distr_prior.sample(350)
posterior_samples = nice.inverse(prior_samples)
plot_true_sampled(x, posterior_samples, SAVE_PATH, "True vs Model Samples - Untrained")
xyi = nice.inverse(xy)
llxy = nice.distr_prior.log_prob(xy)
samples_plot(
    xy,
    prior_samples,
    xyi,
    posterior_samples,
    llxy,
    save_path=SAVE_PATH,
    name="Samples from Un-Trained Model"
)

# Training
nice.fit(x, epochs=850)
# Trained Samples
posterior_samples_trained = nice.inverse(prior_samples)
xyi_trained = nice.inverse(xy)
llxy_trained = nice.distr_prior.log_prob(xy)
plot_true_sampled(x, posterior_samples_trained, SAVE_PATH, "True vs Model Samples - Trained")

samples_plot(
    xy,
    prior_samples,
    xyi_trained,
    posterior_samples_trained,
    llxy_trained,
    save_path=SAVE_PATH,
    name="Samples from Trained Model"
)

plt.show()
