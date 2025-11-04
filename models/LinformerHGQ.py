import tensorflow as tf
from tensorflow.keras import layers, Model


# HGQ imports are optional; raise a clear error if not available
try:
    # Adjust these imports to match your HGQ package
    # Common patterns (one of these should work in your environment):
    # from hgq.keras import QuantizerConfigScope, QEinsumDenseBatchnorm, QAveragePooling1D, QGlobalAveragePooling1D, QAdd
    # from quantizers.keras import QuantizerConfigScope, QEinsumDenseBatchnorm, QAveragePooling1D, QGlobalAveragePooling1D, QAdd
    from hgq.keras import (
        QuantizerConfigScope,
        QEinsumDenseBatchnorm,
        QAveragePooling1D,
        QGlobalAveragePooling1D,
        QAdd,
    )
except Exception as e:  # noqa: F841
    QuantizerConfigScope = None
    QEinsumDenseBatchnorm = None
    QAveragePooling1D = None
    QGlobalAveragePooling1D = None
    QAdd = None


def build_linformer_transformer_classifier_hgq(
    num_particles,
    feature_dim,
    output_dim=5,
    hidden_channels=64,
):
    """
    HGQ-based classifier roughly matching the provided example architecture.

    Inputs shape: (batch, num_particles, feature_dim)
    Outputs shape: (batch, output_dim)
    """
    if any(
        x is None
        for x in [
            QuantizerConfigScope,
            QEinsumDenseBatchnorm,
            QAveragePooling1D,
            QGlobalAveragePooling1D,
            QAdd,
        ]
    ):
        raise ImportError(
            "HGQ layers not found. Please install/import HGQ (e.g., 'hgq.keras') so that "
            "QuantizerConfigScope, QEinsumDenseBatchnorm, QAveragePooling1D, "
            "QGlobalAveragePooling1D, and QAdd are available."
        )

    N = num_particles
    n = feature_dim

    with QuantizerConfigScope(place=("weight", "bias"), b0=4, i0=2, k0=1), QuantizerConfigScope(
        place="datalane", f0=3
    ):
        inp = layers.Input((N, n))

        # Project per-constituent features
        x = QEinsumDenseBatchnorm(
            "bnc,cC->bnC", (N, hidden_channels), bias_axes="C", activation="relu"
        )(inp)

        # S and D branches with residual add
        s = QEinsumDenseBatchnorm(
            "bnc,cC->bnC", (N, hidden_channels), bias_axes="C"
        )(x)
        qs = QAveragePooling1D(pool_size=N)(x)
        d = QEinsumDenseBatchnorm(
            "bnc,cC->bnC", (1, hidden_channels), bias_axes="C"
        )(qs)
        x = layers.ReLU()(QAdd()([s, d]))

        # Another projection + sequence pooling to vector
        x = QEinsumDenseBatchnorm(
            "bnc,cC->bnC", (N, hidden_channels), bias_axes="C", activation="relu"
        )(x)
        x = layers.Flatten()(QGlobalAveragePooling1D()(x))

        # MLP head
        x = QEinsumDenseBatchnorm(
            "bc,cC->bC", hidden_channels, bias_axes="C", activation="relu"
        )(x)
        x = QEinsumDenseBatchnorm("bc,cC->bC", 32, bias_axes="C", activation="relu")(x)
        x = QEinsumDenseBatchnorm("bc,cC->bC", 16, bias_axes="C", activation="relu")(x)

        # Output head with activation consistent with repo losses
        if output_dim == 1:
            out = QEinsumDenseBatchnorm(
                "bc,cC->bC", 1, bias_axes="C", activation="sigmoid"
            )(x)
        else:
            out = QEinsumDenseBatchnorm(
                "bc,cC->bC", output_dim, bias_axes="C", activation="softmax"
            )(x)

        return Model(inp, out)





