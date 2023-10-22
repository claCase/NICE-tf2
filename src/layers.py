import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow_probability.python.bijectors import Bijector


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

    def _forward_log_det_jacobian(
            self,
            y,
            event_ndims=None,
            name="additive_coupling_inverse_log_det_jacobian",
            **kwargs
    ):
        return 0.0


class AffineCoupling(Bijector):
    def __init__(self, scale: Dense, translate: Dense, even: bool, validate_args=False, name="affine_coupling"):
        super().__init__(validate_args, forward_min_event_ndims=0, name=name)
        self.scale = scale
        self.translate = translate
        self.even = even

    def _forward(self, x):
        x1, x2 = x[0],  x[1]
        if not self.even:
            x1, x2 = x2, x1
        h1 = x1
        h2 = x2 * tf.exp(self.scale(x1)) + self.translate(x1)
        return [h1, h2]

    def _inverse(self, h):
        h1, h2 = h[0], h[1]
        x1 = h1
        x2 = (h2 - self.translate(h1))*tf.exp(-self.scale(h1))
        if not self.even:
            x1, x2 = x2, x1
        return [x1, x2]

    def _forward_log_det_jacobian(self, x, event_ndims=None, name="affine_coupling_log_det_jacobian", **kwargs):
        x1, x2 = x[0], x[1]
        return tf.reduce_sum(self.scale(x1))


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

    def _forward_log_det_jacobian(
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

    def inverse_log_det_jacobian(self):
        return self.bijector.inverse_log_det_jacobian(0)
    def forward(self, x):
        return self.bijector.forward(x)

    def inverse(self, x):
        return self.bijector.inverse(x)
