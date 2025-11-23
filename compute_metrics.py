import time

from datasets import load_dataset
import keras
from main import *


dataset = load_dataset("speech-uk/voice-of-america", split='test', streaming=True)
midsize_wav2vec = keras.models.load_model('./final_model.keras', custom_objects={"CTCLossModel": CTCLossModel})
transcribe_using_midsize_wav2vec(midsize_wav2vec, audio)


def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m]


def compute_cer(refs, hyps):
    total_dist = 0
    total_chars = 0

    for r, h in zip(refs, hyps):
        total_dist += levenshtein(r, h)
        total_chars += len(r)

    return total_dist / total_chars if total_chars > 0 else 0


def compute_wer(refs, hyps):
    total_dist = 0
    total_words = 0

    for r, h in zip(refs, hyps):
        r_words = r.split()
        h_words = h.split()

        total_dist += levenshtein(r_words, h_words)
        total_words += len(r_words)

    return total_dist / total_words if total_words > 0 else 0


def compute_ser(refs, hyps):
    sentence_errors = 0
    total_sentences = len(refs)

    for r, h in zip(refs, hyps):
        if r.strip() == "":
            continue
        if levenshtein(r.split(), h.split()) != 0:
            sentence_errors += 1

    return sentence_errors / total_sentences if total_sentences > 0 else 0


if __name__ == "__main__":
    cers = []
    wers = []
    sers = []
    rtfs = []
    for element in dataset:
        ref_text = element['transcription']
        start_time = time.perf_counter()
        hyp_text = transcribe_using_midsize_wav2vec(element['audio']['array'])
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        cers.append(compute_cer(ref_text, hyp_text))
        wers.append(compute_wer(ref_text, hyp_text))
        sers.append(compute_ser(ref_text, hyp_text))
        rtfs.append(elapsed_time / element['duration'])

    print("CER:", sum(cers) / len(cers))
    print("WER:", sum(wers) / len(wers))
    print("SER:", sum(sers) / len(sers))
    print("RTF:", sum(rtfs) / len(rtfs))
