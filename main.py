import tensorflow as tf
from datasets import load_dataset
import string

VOCAB = list(" " + "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя" + "'")  # space must be index 0 in CTC
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

BATCH_SIZE = 8


def extract_features(waveform, sample_rate=16000):
    stfts = tf.signal.stft(waveform, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stfts)

    num_mel_bins = 64
    lower_edge_hertz, upper_edge_hertz = 80.0, sample_rate / 2
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        spectrogram.shape[-1],
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz)

    linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix,
                                          dtype=spectrogram.dtype)

    mel_spectrogram = tf.tensordot(spectrogram,
                                   linear_to_mel_weight_matrix,
                                   1)

    return tf.math.log(mel_spectrogram + 1e-6)


def encode_text(text):
    return [char2idx[c] for c in text.lower() if c in char2idx]


def build_ctc_model(num_classes):
    inputs = tf.keras.Input(shape=(None, 64))  # (time_steps, mel_bins)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(num_classes + 1)(x)  # +1 for the CTC 'blank' token

    return tf.keras.Model(inputs=inputs, outputs=x)


def ctc_batch_cost(y_true, y_pred, input_lengths, label_lengths):
    """
    y_true: int32 tensor of shape (batch_size, max_label_length)
    y_pred: float32 tensor of shape (batch_size, max_time, num_classes + 1) — logits
    input_lengths: int32 tensor of shape (batch_size,) — actual input sequence lengths
    label_lengths: int32 tensor of shape (batch_size,) — actual label lengths
    """
    indices = tf.where(tf.not_equal(y_true, -1))
    values = tf.gather_nd(y_true, indices)
    shape = tf.shape(y_true, out_type=tf.int64)

    sparse_labels = tf.SparseTensor(indices=indices, values=values, dense_shape=tf.cast(shape, tf.int64))

    y_pred = tf.transpose(y_pred, [1, 0, 2])

    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred,
        label_length=None,
        logit_length=input_lengths,
        blank_index=-1
    )

    return loss


class CTCLossModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        features = data["features"]
        labels = data["label"]
        input_lengths = data["input_length"]
        label_lengths = data["label_length"]
        # for label, label_length in zip(labels, label_lengths):
        #     print("label:", label)
        #     print("label_length:", label_length)

        with tf.GradientTape() as tape:
            logits = self.base(features, training=True)
            input_lengths = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
            loss = ctc_batch_cost(labels, logits, input_lengths, label_lengths)

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def decode_ctc_predictions(logits):
    decoded, _ = tf.keras.backend.ctc_decode(logits, input_length=tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1]))
    return decoded[0]


def decode_text(encoded: list[int]) -> str:
    return "".join(idx2char[i] for i in encoded)


def data_generator(raw_dataset):
    for example in raw_dataset:
        text = example["transcription"].strip().lower()
        label = encode_text(text)
        if len(label) == 0:
            continue
        waveform = tf.convert_to_tensor(example["audio"]["array"], dtype=tf.float32)
        log_mel = extract_features(waveform, sample_rate=example["audio"]["sampling_rate"])
        label = encode_text(example["transcription"])
        yield log_mel, label


def prepare_dataset(raw_dataset):
    # output_types = (tf.float32, tf.int32)
    # output_shapes = ([None, 64], [None])  # variable spectrogram and label lengths

    ds = tf.data.Dataset.from_generator(lambda: data_generator(raw_dataset),
                                        output_signature=(
                                            tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                                            tf.TensorSpec(shape=(None,), dtype=tf.int32)))

    def pad_batch(log_mel, label):
        return {
            "features": log_mel,
            "label": label,
            "input_length": tf.shape(log_mel)[0],
            "label_length": tf.shape(label)[0]
        }

    ds = ds.map(pad_batch)
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes={
            "features": [None, 64],
            "label": [None],
            "input_length": [],
            "label_length": []
        },
        padding_values={
            "features": 0.0,
            "label": tf.cast(-1, tf.int32),  # ignored by ctc_batch_cost
            "input_length": 0,
            "label_length": 0
        }
    )

    return ds.prefetch(tf.data.AUTOTUNE)


def train(model: CTCLossModel, dataset: tf.data.Dataset, epochs: int):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.reset_metrics()

        for step, batch in enumerate(dataset):
            logs = model.train_step(batch)
            if step % 10 == 0:
                print(f"  Step {step}: Loss = {logs['loss']:.4f}")

        print(f"Epoch {epoch + 1} completed. Avg Loss: {logs['loss']:.4f}")


"""
Dataset element structure:
{
    audio: {
        path: str,
        array: np.ndarray,
        sampling_rate: int,
    },
    duration: float,
    transcription: str,
}
"""
dataset = load_dataset("speech-uk/voice-of-america", split='train', streaming=True)

prepared_ds = prepare_dataset(dataset)
base_model = build_ctc_model(num_classes=len(VOCAB))
ctc_model = CTCLossModel(base_model)
train(ctc_model, prepared_ds, epochs=10)
