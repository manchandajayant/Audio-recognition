import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max

"""Implementation of peak picking that did not work"""


def generate_constellation(filepath):
    y, sr = librosa.load(filepath, sr=22500)

    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512, window="hann"))

    S_db = librosa.amplitude_to_db(S, ref=np.max, top_db=80)

    coordinates = peak_local_max(S_db, min_distance=10, threshold_abs=-40)
    coordinates = [(t, f) for (f, t) in coordinates]

    return coordinates, S_db


def evaluate_topk_metrics(predicted_tracks, ground_truths, k=3):
    """
    Evaluate recall at k for a list of predicted tracks against ground truth.
    Assumes there is only one relevant (ground truth) file per query.

    Input:
        predicted_tracks (list): List of predicted track filenames (ranked).
        ground_truths (list): List containing the single ground truth track filename.
        k (int): Top-k cutoff (default is 3).

    Output:
        int: recall_value, 1 if any of the top k predictions matches the ground truth, 0 otherwise.
    """
    recall_value = (
        1 if any(track in ground_truths for track in predicted_tracks[:k]) else 0
    )
    return recall_value
