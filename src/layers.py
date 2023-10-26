import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow_probability.python.bijectors import Bijector
from .utils import split_mode, recombine_mode


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
            x,
            event_ndims=None,
            name="additive_coupling_inverse_log_det_jacobian",
            **kwargs
    ):
        return tf.zeros(1)


class AffineCoupling(Bijector):
    def __init__(
            self,
            scale: Dense,
            translate: Dense,
            even: bool,
            validate_args=False,
            name="affine_coupling",
    ):
        super().__init__(validate_args, forward_min_event_ndims=0, name=name)
        self.scale = scale
        self.translate = translate
        self.even = even

    def _forward(self, x):
        x1, x2 = x[0], x[1]
        if not self.even:
            x1, x2 = x2, x1
        h1 = x1
        h2 = x2 * tf.exp(self.scale(x1)) + self.translate(x1)
        return [h1, h2]

    def _inverse(self, h):
        h1, h2 = h[0], h[1]
        x1 = h1
        x2 = (h2 - self.translate(h1)) * tf.exp(-self.scale(h1))
        if not self.even:
            x1, x2 = x2, x1
        return [x1, x2]

    def _forward_log_det_jacobian(
            self, x, event_ndims=None, name="affine_coupling_log_det_jacobian", **kwargs
    ):
        x1, x2 = x[0], x[1]
        return tf.reduce_sum(self.scale(x1), -1)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))


class ExpDiagScaling(Bijector):
    def __init__(self, scale, name="exp_scaling"):
        super().__init__(
            validate_args=False, forward_min_event_ndims=0, name=name
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

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))


class ExpDiagScalingLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(*kwargs)

    def build(self, input_shape):
        if not self.built:
            scale = self.add_weight(shape=(input_shape[-1],), trainable=True, name="scale")
            self.bijector = ExpDiagScaling(scale)
            self.built = True
        else:
            pass

    def call(self, inputs, inverse=False, *args, **kwargs):
        if inverse:
            return self.bijector.inverse(inputs)
        else:
            return self.bijector.forward(inputs)

    def forward_log_det_jacobian(self, x):
        if not self.built:
            self.build(x.shape)
        return self.bijector.forward_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y):
        if not self.built:
            self.build(y.shape)
        return self.bijector.inverse_log_det_jacobian(y)

    def forward(self, x):
        if not self.built:
            self.build(x.shape)
        return self.bijector.forward(x)

    def inverse(self, x):
        if not self.built:
            self.build(x.shape)
        return self.bijector.inverse(x)


class EvenOddBijector(Bijector):
    def __init__(self, name="even_odd_bijector"):
        super().__init__(validate_args=False, forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return split_mode(x, "even_odd")

    def _inverse(self, y):
        return recombine_mode(y, "even_odd")

    def _forward_log_det_jacobian(self, x, event_ndims=None, name="fldj_even_odd_bijector", **kwargs):
        return tf.zeros(1)


class InverseEvenOddBijector(Bijector):
    def __init__(self, name="inverse_even_odd_bijector"):
        super().__init__(validate_args=False, forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return recombine_mode(x, "even_odd")

    def _inverse(self, y):
        return split_mode(y, "even_odd")

    def _forward_log_det_jacobian(self, x, event_ndims=None, name="fldj_inverse_even_odd_bijector", **kwargs):
        return tf.zeros(1)


class SplitBijector(Bijector):
    def __init__(self, name="split_bijector"):
        super().__init__(validate_args=False, forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return split_mode(x, "split")

    def _inverse(self, y):
        return recombine_mode(y, "split")

    def _forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
        return tf.zeros(1)


class InverseSplitBijector(Bijector):
    def __init__(self, name="split_bijector"):
        super().__init__(validate_args=False, forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return recombine_mode(x, "split")

    def _inverse(self, y):
        return split_mode(y, "split")

    def _forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
        return tf.zeros(1)

