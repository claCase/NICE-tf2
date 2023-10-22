import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from typing import Tuple
from .layers import AdditiveCoupling, ExpDiagScalingLayer
from tensorflow_probability.python.distributions import Normal, Logistic, Independent


class NICE(Model):
    def __init__(
            self,
            output_dim: int = 2,
            n_couplings: int = 4,
            hidden_units: Tuple = (10, 5),
            mode="split",
            activation="relu",
            distr="gaussian",
            name="nice",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        assert distr in ["gaussian", "logistic"], "distribution not supported"
        self.output_dim = output_dim
        self.mode = mode
        assert mode in ["split", "even_odd"]
        self.transforms = []
        for i in range(n_couplings):
            layers = []
            for unit in hidden_units:
                layers.append(Dense(unit, activation))
            layers.append(Dense(output_dim // 2, activation="linear"))
            coupling = AdditiveCoupling(
                tf.keras.models.Sequential(layers), even=bool(i % 2)
            )
            self.transforms.append(coupling)
        self.scaling = ExpDiagScalingLayer()
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

    def forward(self, inputs):
        x = inputs
        x = self.split_mode(x)
        for i, coupling in enumerate(self.transforms):
            x = coupling.forward(x)
        x = self.recombine_mode(x)
        return self.scaling(x)

    def inverse(self, inputs):
        h = self.scaling(inputs, inverse=True)
        h = self.split_mode(h)
        for i, coupling in enumerate(reversed(self.transforms)):
            h = coupling.inverse(h)
        h = self.recombine_mode(h)
        return h

    def call(self, inputs, inverse=False, training=None, mask=None):
        if inverse:
            return self.inverse(inputs)
        else:
            return self.forward(inputs)

    def loss_fn(self, x):
        probs, log_sum_det = self.log_prob(x)
        return tf.reduce_mean(probs) + log_sum_det

    def log_prob(self, x):
        h = self.call(x)
        log_prob_prior = self.distr_prior.log_prob(h)
        log_det_jac = self.scaling.forward_log_det_jacobian()
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
        if self.mode == "split":
            h1, h2 = tf.split(inputs, 2, axis=-1)
        else:
            h1, h2 = inputs[..., ::2], inputs[..., 1::2]
        return h1, h2

    def recombine_mode(self, inputs):
        a, b = inputs
        x, y = tf.shape(a)[0], tf.shape(a)[1]
        if self.mode == "split":
            return tf.concat([a, b], -1)
        else:
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

    def inpainting(self, masked_inputs, mask, steps):
        def alpha(i):
            return 10 / (100 + i)

        assert masked_inputs.shape == mask.shape
        if not isinstance(masked_inputs, tf.Tensor):
            masked_inputs = tf.convert_to_tensor(masked_inputs, tf.float32)

        if not isinstance(mask, tf.Tensor):
            masked_inputs = tf.convert_to_tensor(mask, tf.float32)

        for i in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(masked_inputs)
                ll, _ = self.log_prob(masked_inputs)
            grads = tape.gradient(ll, masked_inputs)
            masked_inputs = masked_inputs + alpha(i) * (grads + tf.random.normal(shape=masked_inputs.shape)) * mask
        return masked_inputs
