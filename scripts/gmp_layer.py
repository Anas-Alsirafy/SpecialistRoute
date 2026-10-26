# ═══════════════════════════════════════════════════════════════════════════════
# GENERALIZED MEAN POOLING (GMP) — drop-in replacement for GlobalAveragePooling2D
#
# GMP(x) = ( mean_i( clip(x_i, eps, inf)^rho ) ) ^ (1/rho)
#   rho = 1        -> equivalent to average pooling
#   rho -> infinity -> approaches max pooling
#   rho is LEARNABLE, initialized to 3.0 (per Arshad et al. 2026, ConvGeM-NeXt)
#
# TWO NUMERICAL SAFEGUARDS not mentioned in the paper's description, but required
# for this to actually work on a real network:
#
#   1. CLIPPING TO NON-NEGATIVE: x^rho for fractional rho is undefined for
#      negative x (produces NaN). ConvNeXt blocks use GELU activations, which
#      CAN produce negative outputs -- unlike the ReLU-based networks GMP was
#      originally developed for in image retrieval (Radenovic et al.), where
#      features are non-negative by construction. Values are clipped to
#      [eps, +inf) before the power operation. This is standard practice in
#      real GMP implementations, not an approximation of the paper's method --
#      it is what makes the paper's own formula well-defined on this backbone.
#
#   2. CONSTRAINING RHO TO STAY POSITIVE: an unconstrained learnable rho could
#      drift to zero or negative during training via ordinary gradient descent,
#      which makes the whole operation undefined or degenerate. rho is
#      reparameterized as rho = softplus(raw_rho) + min_rho, which keeps it
#      strictly positive (minimum min_rho) for any real-valued raw_rho, so no
#      clipping/projection step is needed after each optimizer update.
# ═══════════════════════════════════════════════════════════════════════════════

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="custom")
class GeneralizedMeanPooling2D(tf.keras.layers.Layer):
    def __init__(self, init_rho=3.0, min_rho=1.0, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_rho = init_rho
        self.min_rho = min_rho
        self.eps = eps
        # softplus(raw_rho_init) + min_rho = init_rho  =>  solve for raw_rho_init
        target = init_rho - min_rho
        self._raw_rho_init = float(target + tf.math.log(1.0 - tf.exp(-tf.cast(target, tf.float32))).numpy()) \
            if target > 0 else 0.0

    def build(self, input_shape):
        self.raw_rho = self.add_weight(
            name="raw_rho", shape=(), dtype=self.dtype,
            initializer=tf.keras.initializers.Constant(self._raw_rho_init),
            trainable=True,
        )
        super().build(input_shape)

    @property
    def rho(self):
        return tf.nn.softplus(self.raw_rho) + self.min_rho

    def call(self, inputs):
        x = tf.clip_by_value(inputs, self.eps, tf.float32.max)
        rho = self.rho
        pooled = tf.reduce_mean(tf.pow(x, rho), axis=[1, 2])
        return tf.pow(pooled, 1.0 / rho)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"init_rho": self.init_rho, "min_rho": self.min_rho, "eps": self.eps})
        return config
