import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

wav_path = "audio/cut_test.mp3"

y, sr = librosa.load(wav_path, sr=None)

S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    power=2.0
)

S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 5))
librosa.display.specshow(
    S_dB,
    sr=sr,
    hop_length=512,
    x_axis='time',
    y_axis='mel',
    cmap='magma'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()
