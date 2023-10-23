import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from typing import Tuple
from .layers import AdditiveCoupling, ExpDiagScalingLayer, AffineCoupling
from tensorflow_probability.python.distributions import Normal, Logistic, Independent
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
        self.scaling = None
        self.build_coupling_transforms()

    @abstractmethod
    def build_coupling_transforms(self):
        raise NotImplementedError

    def forward(self, inputs):
        if self.transforms is None or self.scaling is None:
            raise NotImplementedError("Implement build_coupling_transforms method")
        x = inputs
        x = self.split_mode(x)
        log_det_jac = 0.0
        for i, coupling in enumerate(self.transforms):
            x = coupling.forward(x)
            log_det_jac = log_det_jac + coupling.forward_log_det_jacobian(x)
        x = self.recombine_mode(x)
        return self.scaling(x), log_det_jac + self.scaling.forward_log_det_jacobian(x)

    def inverse(self, inputs):
        if self.transforms is None or self.scaling is None:
            raise NotImplementedError("Implement build_coupling_transforms method")
        h = self.scaling(inputs, inverse=True)
        h = self.split_mode(h)
        log_det_jac = self.scaling.inverse_log_det_jacobian(inputs)
        for i, coupling in enumerate(reversed(self.transforms)):
            h = coupling.inverse(h)
            log_det_jac = log_det_jac + coupling.inverse_log_det_jacobian(h)
        h = self.recombine_mode(h)
        return h, log_det_jac

    def call(self, inputs, inverse=False, training=None, mask=None):
        if inverse:
            return self.inverse(inputs)[0]
        else:
            return self.forward(inputs)[0]

    def loss_fn(self, x):
        probs, log_sum_det = self.log_prob(x)
        return tf.reduce_mean(probs + log_sum_det)

    def log_prob(self, x):
        h, log_det_jac = self.forward(x)
        log_prob_prior = self.distr_prior.log_prob(h)
        return log_prob_prior, log_det_jac

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = -self.loss_fn(data)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"LogLoss": self.loss_tracker.result()}

    def sample(self, n_samples):
        h = self.distr_prior.sample(n_samples)
        return self.call(h, inverse=True)

    def split_mode(self, inputs):
        return split_mode(inputs, self.mode)

    def recombine_mode(self, inputs):
        return recombine_mode(inputs, self.mode)


class NICEFlow(Flow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_coupling_transforms(self):
        self.transforms = []
        for i in range(self.n_couplings):
            layers = []
            for unit in self.hidden_units:
                layers.append(Dense(unit, self.activation))
            layers.append(Dense(self.output_dim // 2, activation="linear"))
            coupling = AdditiveCoupling(
                tf.keras.models.Sequential(layers), even=bool(i % 2)
            )
            self.transforms.append(coupling)
        self.scaling = ExpDiagScalingLayer()


class RealNVPFlow(Flow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_coupling_transforms(self):
        self.transforms = []
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
        self.scaling = ExpDiagScalingLayer()
