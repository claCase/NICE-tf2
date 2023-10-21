from typing import List, Tuple
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.models import Model
from tensorflow_probability.python.bijectors import Bijector
from tensorflow_probability.python.distributions import Normal, Logistic, Independent


class AdditiveCoupling(Bijector):
    def __init__(
        self, mlp: Dense, even: bool, validate_args=False, name="additive_coupling"
    ):
        super(AdditiveCoupling, self).__init__(
            validate_args, forward_min_event_ndims=0, name=name
        )
        self.mlp = mlp
        self.even = even

    def _forward(self, x):
        x1, x2 = x[0], x[1]
        if not self.even:
            x1, x2 = x2, x1
        h1 = x1
        h2 = x2 + self.mlp(x1)
        return [h1, h2]

    def _inverse(self, h):
        h1, h2 = h[0], h[1]
        x1 = h1
        x2 = h2 - self.mlp(h1)
        if not self.even:
            x1, x2 = x2, x1
        return [x1, x2]

    def _inverse_log_det_jacobian(
        self,
        y,
        event_ndims=None,
        name="additive_coupling_inverse_log_det_jacobian",
        **kwargs
    ):
        return 0.0


class ExpDiagScaling(Bijector):
    def __init__(self, scale, **kwargs):
        super().__init__(
            validate_args=False, forward_min_event_ndims=0, name=kwargs.get("name")
        )
        self.scale = scale

    def _forward(self, x):
        return x * tf.exp(self.scale)

    def _inverse(self, y):
        return y * tf.exp(-self.scale)

    def forward_log_det_jacobian(
        self, x, event_ndims=None, name="forward_log_det_jacobian", **kwargs
    ):
        return tf.reduce_sum(self.scale)


class ExpDiagScalingLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(*kwargs)

    def build(self, input_shape):
        scale = self.add_weight(shape=(input_shape[-1],), trainable=True)
        self.bijector = ExpDiagScaling(scale)

    def call(self, inputs, inverse=False, *args, **kwargs):
        if inverse:
            return self.bijector.inverse(inputs)
        else:
            return self.bijector.forward(inputs)

    def forward_log_det_jacobian(self):
        return self.bijector.forward_log_det_jacobian(0)

    def forward(self, x):
        return self.bijector.forward(x)

    def inverse(self, x):
        return self.bijector.inverse(x)


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
