import tensorflow as tf
from pathlib import Path
from typing import Iterator, List
import subprocess


def iter_flac_files(root: str | Path) -> Iterator[Path]:
    """
    Walk *root* breadth‑first and yield FLAC files that satisfy:

        <split>/LibriSpeech/<split>/<speaker>/<chapter>/<file>.flac

    Rules
    -----
    • While descending, discard anything that is **not a directory** until you
      reach a leaf directory (one that contains no sub‑directories).
    • In a leaf directory, yield only files whose suffix is '.flac'
      (case‑insensitive).

    Parameters
    ----------
    root : str | pathlib.Path
        The starting folder – e.g. ``"dev-clean/LibriSpeech/dev-clean"``.

    Yields
    ------
    pathlib.Path
        Absolute or relative paths (depending on *root*) to every FLAC file.
    """
    root = Path(root)
    queue: List[Path] = [root]

    while queue:
        current = queue.pop()

        # Separate children into dirs and files first – we must know
        # if *current* is a leaf before acting on its files.
        dirs = [p for p in current.iterdir() if p.is_dir()]
        files = [p for p in current.iterdir() if p.is_file()]

        if dirs:                     # **not** at the final level yet
            queue.extend(dirs)       # …so follow sub‑directories only
        else:                        # leaf directory → keep *.flac files
            for f in files:
                if f.suffix.lower() == ".flac":
                    yield f


def decode_flac_ffmpeg(path: tf.Tensor, sample_rate: int = 16000, channels: int = 1) -> tf.Tensor:
    """
    Decode FLAC file to PCM audio using FFmpeg via subprocess.

    Parameters
    ----------
    path : tf.Tensor of type string
        File path to a .flac file.
    sample_rate : int
        Desired sample rate (default 16000).
    channels : int
        Number of audio channels (default 1 = mono).

    Returns
    -------
    tf.Tensor
        A float32 tensor containing the waveform normalized between [-1, 1].
    """
    def _decode(t: tf.Tensor) -> tf.Tensor:
        # path = path_str.decode("utf-8")
        path_str = t.numpy().decode("utf-8")
        command = [
            "ffmpeg",
            "-i", path_str,
            "-f", "s16le",             # raw 16-bit PCM
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-loglevel", "error",
            "-"
        ]
        try:
            raw_audio = subprocess.check_output(command)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed on file: {path_str}") from e

        audio = tf.io.decode_raw(raw_audio, tf.int16)
        audio = tf.cast(audio, tf.float32) / 32768.0  # Normalize to [-1, 1]
        return audio

    audio = tf.py_function(func=_decode, inp=[path], Tout=tf.float32)
    return audio


def make_flac_dataset(root: str | Path, desired_channels: int = 1):
    """
    Build a `tf.data.Dataset` whose elements are ``(audio_tensor, sample_rate)``.
    Each tensor is produced by `tf.audio.decode_flac`, so you can feed it
    directly into a model or further preprocessing pipeline.

    Example
    -------
    >>> ds = make_flac_dataset("dev-clean/LibriSpeech/dev-clean")
    >>> audio, sr = next(iter(ds))
    >>> print(audio.shape, sr.numpy())     # (num_samples, 1) 16000
    """

    flac_paths = [str(p) for p in iter_flac_files(root)]
    ds = tf.data.Dataset.from_tensor_slices(flac_paths)

    # TF‐side loader
    def _load(path):
        # raw = tf.io.read_file(path)
        audio = decode_flac_ffmpeg(path)
        return audio, tf.constant(16000, tf.int32)

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == "__main__":
    base = r"data\dev-clean\LibriSpeech\dev-clean"  # or Path("...")

    dataset = make_flac_dataset(base)
    audio, sr = next(iter(dataset))
    print("\nDecoded first sample:", audio.shape, "at", sr.numpy(), "Hz")
