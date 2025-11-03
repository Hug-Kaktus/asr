import tensorflow as tf


def build_wav2vec2_ctc_model(num_classes, model_dim=256, num_layers=6, num_heads=4, ff_dim=1024):
    inputs = tf.keras.Input(shape=(None, 64))

    x = tf.keras.layers.Conv1D(128, 10, strides=5, padding='same', activation='gelu')(inputs)
    x = tf.keras.layers.Conv1D(256, 3, strides=2, padding='same', activation='gelu')(x)
    x = tf.keras.layers.Conv1D(model_dim, 3, strides=2, padding='same', activation='gelu')(x)

    pos_emb = tf.keras.layers.Embedding(1000, model_dim)(tf.range(tf.shape(x)[1]))
    pos_emb = tf.expand_dims(pos_emb, 0)
    x = x + pos_emb[:, :tf.shape(x)[1], :]

    for _ in range(num_layers):
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim//num_heads)(x, x)
        attn_out = tf.keras.layers.Dropout(0.1)(attn_out)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(model_dim),
            tf.keras.layers.Dropout(0.1)
        ])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))

    x = tf.keras.layers.Dense(num_classes + 1, name="ctc_logits")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="wav2vec2_ctc")
