import os

REF_DIR = "transcriber/static/references"
HYP_DIR = "transcriber/static/hypotheses"


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


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
    reference_files = sorted([f for f in os.listdir(REF_DIR)])
    hypothesis_files = sorted([f for f in os.listdir(HYP_DIR)])
    common_files = sorted(set(reference_files) & set(hypothesis_files))
    for filename in common_files:
        ref_text = load_text(os.path.join(REF_DIR, filename))
        hyp_text = load_text(os.path.join(HYP_DIR, filename))

        cers.append(compute_cer(ref_text, hyp_text))
        wers.append(compute_wer(ref_text, hyp_text))
        sers.append(compute_ser(ref_text, hyp_text))

    common_files = sorted(set(reference_files) & set(hypothesis_files))
    print("CER:", sum(cers) / len(cers))
    print("WER:", sum(wers) / len(wers))
    print("SER:", sum(sers) / len(sers))
