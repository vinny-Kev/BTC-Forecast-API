"""Custom Transformer components required for loading the v2 regression model."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="CustomLayers", name="PositionalEncoding")
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding layer compatible with the saved transformer model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sequence_length: int | None = None
        self._d_model: int | None = None
        self._positional_encoding: tf.Tensor | None = None

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        super().build(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                "PositionalEncoding expects inputs with shape (batch, sequence_length, d_model)"
            )

        self._sequence_length = int(input_shape[1])
        self._d_model = int(input_shape[2])

        position = tf.range(self._sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self._d_model, 2, dtype=tf.float32)
            * -(math.log(10000.0) / self._d_model)
        )

        angle_rads = position * div_term
        # Interleave sin and cos for even/odd indices
        sin_terms = tf.sin(angle_rads)
        cos_terms = tf.cos(angle_rads)

        pos_encoding = tf.reshape(
            tf.stack([sin_terms, cos_terms], axis=-1),
            (self._sequence_length, self._d_model),
        )

        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        self._positional_encoding = tf.cast(pos_encoding, dtype=self.compute_dtype)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self._positional_encoding is None:
            raise RuntimeError("PositionalEncoding layer must be built before calling")

        seq_len = tf.shape(inputs)[1]
        return inputs + self._positional_encoding[:, :seq_len, :]

    def get_config(self) -> Dict[str, Any]:  # type: ignore[override]
        config = super().get_config()
        config.update(
            {
                "sequence_length": self._sequence_length,
                "d_model": self._d_model,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PositionalEncoding":
        # sequence_length and d_model are derived during build, so pop if present
        config.pop("sequence_length", None)
        config.pop("d_model", None)
        return cls(**config)


@register_keras_serializable(package="CustomLayers", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    """Standard Transformer encoder block with residual connections."""

    def __init__(
        self,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        key_dim: Optional[int] = None,
        d_model: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.key_dim = key_dim
        self.d_model = d_model

        # We'll create a nested attention "att" sublayer and a nested "ffn"
        # sequential so that the saved variable groups (e.g. att/key_dense)
        # match exactly when loading weights.
        self.att = None
        self._attention_output_dense = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        super().build(input_shape)
        if len(input_shape) != 3:
            raise ValueError("TransformerBlock expects inputs with shape (batch, sequence_length, d_model)")

        d_model = self.d_model or int(input_shape[-1])
        seq_len = int(input_shape[1])
        # The saved model was exported with key_dim == d_model, so default to
        # that when key_dim is not explicitly provided. This ensures the
        # EinsumDense internal kernel shapes match the saved weights.
        key_dim = self.key_dim or d_model

        # Create attention sublayer with exact child names used by the saved model
        att = layers.Layer(name="att")

        # Note: saved model used small helper layers like "_softmax" and
        # "_dropout_layer" but creating layers with leading underscore can
        # trigger scope/name validation issues; we'll use tf.nn operations
        # for softmax and dropout at call-time instead of creating named
        # sublayers here.

        # query/key/value use EinsumDense producing (batch, seq, num_heads, key_dim)
        att.key_dense = layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_heads, key_dim),
            bias_axes="nh",
            name="key_dense",
        )
        att.query_dense = layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_heads, key_dim),
            bias_axes="nh",
            name="query_dense",
        )
        att.value_dense = layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_heads, key_dim),
            bias_axes="nh",
            name="value_dense",
        )

        # output projection from (batch, seq, num_heads, key_dim) -> (batch, seq, d_model)
        # The saved model stores the kernel with shape (num_heads, key_dim, d_model).
        # Build a small custom Layer that creates weights with that exact shape so
        # weight loading will map variables by name and shape.
        class OutputDense(layers.Layer):
            def __init__(self, num_heads, key_dim, d_model, **kw):
                super().__init__(**kw)
                self._num_heads = num_heads
                self._key_dim = key_dim
                self._d_model = d_model

            def build(self, input_shape):
                # kernel shape matches saved weights: (num_heads, key_dim, d_model)
                self.kernel = self.add_weight(
                    name="kernel",
                    shape=(self._num_heads, self._key_dim, self._d_model),
                    initializer="glorot_uniform",
                    trainable=True,
                )
                self.bias = self.add_weight(name="bias", shape=(self._d_model,), initializer="zeros", trainable=True)

            def call(self, x):
                # x: (batch, seq, num_heads, key_dim)
                return tf.einsum("btnh,nhd->btd", x, self.kernel) + self.bias

        att.output_dense = OutputDense(self.num_heads, key_dim, d_model, name="output_dense")

        self.att = att

        # Feed-forward network: use a Sequential named "ffn" so nested layer naming
        # matches saved structure ffn/layers/dense and ffn/layers/dense_1
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation="relu", name="dense"),
                layers.Dropout(self.dropout, name="dropout"),
                layers.Dense(d_model, name="dense_1"),
            ],
            name="ffn",
        )

        # LayerNorm and outer dropout names to match saved model
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="layernorm1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="layernorm2")
        self.dropout1 = layers.Dropout(self.dropout, name="dropout1")
        self.dropout2 = layers.Dropout(self.dropout, name="dropout2")

        # Build internal layers with explicit shapes so variables are created now
        shape = tf.TensorShape((None, seq_len, d_model))
        attn_shape = tf.TensorShape((None, seq_len, self.num_heads, key_dim))
        ff_shape = tf.TensorShape((None, seq_len, self.ff_dim))

        # Build attention internals
        att.query_dense.build(shape)
        att.key_dense.build(shape)
        att.value_dense.build(shape)
        att.output_dense.build(attn_shape)

        # Build ffn
        self.ffn.build(shape)

        # Build layer norms
        self.layernorm1.build(shape)
        self.layernorm2.build(shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        if self.att is None or self.ffn is None or self.layernorm1 is None or self.layernorm2 is None:
            raise RuntimeError("TransformerBlock layer must be built before calling")

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Multi-head attention via nested att sublayer
        query = self.att.query_dense(inputs)
        key = self.att.key_dense(inputs)
        value = self.att.value_dense(inputs)

        # Transpose to (batch, num_heads, seq_len, key_dim)
        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 1, 3))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        # Attention scores and softmax
        key_dim = tf.shape(query)[-1]
        attention_scores = tf.einsum("bhqd,bhkd->bhqk", query, key)
        attention_scores = attention_scores / tf.sqrt(tf.cast(key_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        if training and self.dropout > 0:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout)

        # Attention output projection
        attention_output = tf.einsum("bhqk,bhvd->bhqd", attention_weights, value)
        # transpose to (batch, seq, num_heads, key_dim) to match saved model
        attention_output = tf.transpose(attention_output, perm=(0, 2, 1, 3))
        attention_output = self.att.output_dense(attention_output)

        # Residual connection and layer norm
        out1 = self.layernorm1(inputs + attention_output)

        # Feed-forward network (Sequential ffn)
        ffn_output = self.ffn(out1, training=training)

        # Residual connection and final norm
        return self.layernorm2(out1 + ffn_output)

    def get_config(self) -> Dict[str, Any]:  # type: ignore[override]
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "key_dim": self.key_dim,
                "d_model": self.d_model,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TransformerBlock":
        return cls(**config)
