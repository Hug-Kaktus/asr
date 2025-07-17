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
