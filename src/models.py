import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from typing import Tuple
from .layers import AdditiveCoupling, ExpDiagScalingLayer, AffineCoupling, EvenOddBijector, InverseEvenOddBijector
from tensorflow_probability.python.distributions import Normal, Logistic, Independent
from tensorflow_probability.python.bijectors import BatchNormalization
from .utils import split_mode, recombine_mode
from abc import abstractmethod


class Flow(Model):
    def __init__(
            self,
            output_dim: int = 2,
            n_couplings: int = 4,
            hidden_units: Tuple = (10, 5),
            mode="even_odd",
            activation="relu",
            distr="logistic",
            name="Flow",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        assert distr in ["gaussian", "logistic"], "distribution not supported"
        self.output_dim = output_dim
        self.mode = mode
        assert mode in ["split", "even_odd"]
        if distr == "gaussian":
            self.distr_prior = Independent(
                Normal(tf.zeros(output_dim), tf.ones(output_dim)),
                reinterpreted_batch_ndims=1,
            )
        else:
            self.distr_prior = Independent(
                Logistic(tf.ones(output_dim), tf.ones(output_dim)),
                reinterpreted_batch_ndims=1,
            )
        self.loss_tracker = tf.keras.metrics.Mean(name="log_prob")
        self.n_couplings = n_couplings
        self.hidden_units = hidden_units
        self.activation = activation
        self.transforms = None

        self.build_coupling_transforms()

    @abstractmethod
    def build_coupling_transforms(self):
        raise NotImplementedError

    @tf.function
    def forward(self, inputs):
        if self.transforms is None:
            raise NotImplementedError("Implement build_coupling_transforms method")
        x = inputs
        log_det_jac = 0.0
        for i, coupling in enumerate(self.transforms):
            x = coupling.forward(x)
            log_det_jac = log_det_jac + coupling.forward_log_det_jacobian(x)
        return x, log_det_jac

    @tf.function
    def inverse(self, inputs):
        if self.transforms is None:
            raise NotImplementedError("Implement build_coupling_transforms method")
        h = inputs
        log_det_jac = 0.
        for i, coupling in enumerate(reversed(self.transforms)):
            h = coupling.inverse(h)
            log_det_jac = log_det_jac + coupling.inverse_log_det_jacobian(h)
        return h, log_det_jac

    def call(self, inputs, inverse=False, training=None, mask=None):
        if inverse:
            return self.inverse(inputs)[0]
        else:
            return self.forward(inputs)[0]

    @tf.function
    def loss_fn(self, x):
        probs, log_sum_det = self.log_prob(x)
        return - tf.reduce_mean(probs + log_sum_det)

    @tf.function
    def log_prob(self, x):
        h, log_det_jac = self.forward(x)
        log_prob_prior = self.distr_prior.log_prob(h)
        return log_prob_prior, log_det_jac

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(data)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"LogLoss": self.loss_tracker.result()}

    @tf.function
    def sample(self, n_samples):
        h = self.distr_prior.sample(n_samples)
        return self.call(h, inverse=True)

    @tf.function
    def inpainting(self, noised_inputs, noised_mask, n_steps: int, noise_scale=0.2):
        """
        Function used for denoising and inpainting by updating the input in the direction of maximum likelihood
        :param noised_inputs: Corrupted Inputs
        :param noised_mask: Binary mask select the corrupted inputs to update
        :param n_steps: how many steps to denoise the input
        :return: denoised input
        """

        def alpha(i):
            return 10 / (100 + i)

        if isinstance(noised_inputs, np.ndarray):
            noised_inputs = tf.convert_to_tensor(noised_inputs, tf.float32)

        for i in range(n_steps):
            with tf.GradientTape() as tape:
                tape.watch(noised_inputs)
                ll = self.loss_fn(noised_inputs)
            grads = tape.gradient(ll, noised_inputs)
            noised_inputs += alpha(i) * (grads + tf.random.normal(shape=grads.shape) * noise_scale) * noised_mask
        return noised_inputs


class NICEFlow(Flow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_coupling_transforms(self):
        self.transforms = []
        self.transforms.append(EvenOddBijector())
        for i in range(self.n_couplings):
            layers = []
            for unit in self.hidden_units:
                layers.append(Dense(unit, self.activation))
            layers.append(Dense(self.output_dim // 2, activation="linear"))
            coupling = AdditiveCoupling(
                tf.keras.models.Sequential(layers), even=bool(i % 2)
            )
            self.transforms.append(coupling)
        self.transforms.append(InverseEvenOddBijector())
        self.transforms.append(ExpDiagScalingLayer())


class RealNVPFlow(Flow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_coupling_transforms(self):
        self.transforms = []
        self.transforms.append(EvenOddBijector())
        for i in range(self.n_couplings):
            scale_layers = []
            translation_layers = []
            for unit in self.hidden_units:
                scale_layers.append(Dense(unit, self.activation))
                translation_layers.append(Dense(unit, self.activation))
            scale_layers.append(Dense(self.output_dim // 2, activation="linear"))
            translation_layers.append(Dense(self.output_dim // 2, activation="linear"))
            coupling = AffineCoupling(
                tf.keras.models.Sequential(scale_layers),
                tf.keras.models.Sequential(translation_layers),
                even=bool(i % 2),
            )
            self.transforms.append(coupling)
            self.transforms.append(InverseEvenOddBijector())
            self.transforms.append(BatchNormalization())
            self.transforms.append(EvenOddBijector())
        self.transforms.append(InverseEvenOddBijector())
