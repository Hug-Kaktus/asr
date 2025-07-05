import tensorflow as tf
from datasets import load_dataset


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

    linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix, dtype=spectrogram.dtype)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)

    # Avoid log of zero
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram


def preprocess(waveform, label):
    features = extract_features(waveform)
    return features, label


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
dataset_element = next(iter(dataset))
print(extract_features(dataset_element['audio']['array']))
