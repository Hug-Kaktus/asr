import tensorflow as tf


def build_ctc_model(num_classes):
    inputs = tf.keras.Input(shape=(None, 64))  # (time_steps, mel_bins)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(num_classes + 1)(x)  # +1 for the CTC 'blank' token

    return tf.keras.Model(inputs=inputs, outputs=x)


def deep_ctc_model(num_classes):
    inputs = tf.keras.Input(shape=(None, 64), name="input")  # (time_steps, mel_bins)
    x = inputs

    # 1. Initial Conv1D stack (deep temporal feature extraction)
    for filters, kernel_size, dropout in [
        (256, 11, 0.1),
        (256, 11, 0.1),
        (384, 13, 0.2),
        (512, 15, 0.2),
    ]:
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    # 3. Recurrent layers (for long-range dependency)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # 4. Final dense projection
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes + 1)(x)  # +1 for CTC blank token

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# class DebugLayer(tf.keras.layers.Layer):
#     def __init__(self, layer, name=None):
#         super().__init__(name=name)
#         self.layer = layer
# 
#     def call(self, inputs, **kwargs):
#         print(f"Input to {self.layer.name}: {inputs.shape}")
#         output = self.layer(inputs, **kwargs)
#         print(f"Output from {self.layer.name}: {output.shape}")
#         return output
# 
# 
# def build_debug_ctc_model(num_classes):
#     inputs = tf.keras.Input(shape=(None, 64))  # (time_steps, mel_bins)
# 
#     x = DebugLayer(tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu', name='conv1d'))(inputs)
#     x = DebugLayer(tf.keras.layers.MaxPooling1D(2, name='maxpool'))(x)
# 
#     x = DebugLayer(tf.keras.layers.Bidirectional(
#         tf.keras.layers.GRU(128, return_sequences=True), name='bidir_gru'))(x)
# 
#     x = DebugLayer(tf.keras.layers.Dropout(0.3, name='dropout'))(x)
#     x = DebugLayer(tf.keras.layers.Dense(num_classes + 1, name='logits'))(x)  # +1 for CTC blank
# 
#     return tf.keras.Model(inputs=inputs, outputs=x)
