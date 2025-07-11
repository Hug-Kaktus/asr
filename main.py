import tensorflow as tf
from datasets import load_dataset
import string

VOCAB = list(" " + string.ascii_lowercase + "'")  # space must be index 0 in CTC
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}


def extract_features(waveform, sample_rate=16000):
    # Convert waveform to spectrogram
    stfts = tf.signal.stft(waveform, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stfts)

    # Convert to mel-scale
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


def ctc_loss(y_true, y_pred, input_lengths, label_lengths):
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_lengths, label_lengths)
    return loss


class CTCLossModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def train_step(self, data):
        x, y_true, input_lengths, label_lengths = data

        with tf.GradientTape() as tape:
            y_pred = self.base(x, training=True)
            loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_lengths, label_lengths)

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        return {"loss": tf.reduce_mean(loss)}


def decode_ctc_predictions(logits):
    decoded, _ = tf.keras.backend.ctc_decode(logits, input_length=tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1]))
    return decoded[0]


def decode_to_text(predictions):
        return ["".join([idx2char[i.numpy()] for i in seq if i.numpy() != -1]) for seq in predictions]

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
# dataset_element = next(iter(dataset))
# print(extract_features(dataset_element['audio']['array']))
