import tensorflow as tf


VOCAB = list(" " + "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя" + "'")  # space must be index 0 in CTC
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

BATCH_SIZE = 8


@tf.function
def extract_features(waveform, sample_rate=16000):
    stfts = tf.signal.stft(waveform, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stfts)

    num_mel_bins = 64
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz = 80.0, sample_rate / 2
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
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
        },
        drop_remainder=True
    )

    return ds.prefetch(tf.data.AUTOTUNE)
