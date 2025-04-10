import json
import os
import pickle
from collections import defaultdict

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

# from math import fma


SR = 22050
N_FFT = 4096
HOP_LENGTH = 512

PEAK_NEIGHBORHOOD_SIZE = (20, 20)
AMP_MIN = -80


T_DELTA_MIN = 0.0
T_DELTA_MAX = 2.0
TIME_RESOLUTION = 0.01

F_MIN = 30
F_MAX = 4000
F_BITS = 10
DT_BITS = 10

import librosa
import numpy as np


def generate_constellation(
    audio_file,
    query=False,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    peak_neighborhood_size=PEAK_NEIGHBORHOOD_SIZE,
    amp_min=AMP_MIN,
):

    y, sr = librosa.load(audio_file, sr=sr, mono=False)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    local_max = maximum_filter(S_db, size=peak_neighborhood_size)
    peaks = (S_db == local_max) & (S_db > amp_min)

    # Get indices of the peaks
    peak_indices = np.argwhere(peaks)
    # Convert indices to time (columns) and frequency (rows)
    times = librosa.frames_to_time(peak_indices[:, 1], sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[peak_indices[:, 0]]

    # Combine into an array of (time, frequency)
    peaks_coords = np.vstack((times, freqs)).T
    return peaks_coords, S_db


# def pack_hash(
#     f1,
#     f2,
#     dt,
#     f_min=F_MIN,
#     f_max=F_MAX,
#     f_bits=F_BITS,
#     dt_bits=DT_BITS,
#     time_res=TIME_RESOLUTION,
# ):

#     if f1 < f_min or f1 > f_max or f2 < f_min or f2 > f_max:
#         return None

#     q_level = (2**f_bits) - 1
#     q1 = int((f1 - f_min) / (f_max - f_min) * q_level)
#     q2 = int((f2 - f_min) / (f_max - f_min) * q_level)

#     dt_int = int(dt / time_res)
#     if dt_int >= (2**dt_bits):
#         return None

#     h = (q1 << (dt_bits + f_bits)) | (q2 << dt_bits) | dt_int
#     return h


def pack_hash(q1, q2, dt_int, f_bits=F_BITS, dt_bits=DT_BITS):
    if dt_int >= (2**dt_bits):
        return None
    return (q1 << (dt_bits + f_bits)) | (q2 << dt_bits) | dt_int


def generate_hashes(
    constellation,
    t_delta_min=T_DELTA_MIN,
    t_delta_max=T_DELTA_MAX,
    f_min=F_MIN,
    f_max=F_MAX,
    f_bits=F_BITS,
    dt_bits=DT_BITS,
    time_res=TIME_RESOLUTION,
):
    hashes = []
    constellation_sorted = sorted(constellation, key=lambda x: x[0])
    n_points = len(constellation_sorted)
    q_level = (2**f_bits) - 1

    for i, anchor in enumerate(constellation_sorted):
        t_anchor, f_anchor = anchor

        # Skip if frequency is outside bounds
        if f_anchor < f_min or f_anchor > f_max:
            continue

        q1 = int((f_anchor - f_min) / (f_max - f_min) * q_level)

        for j in range(i + 1, n_points):
            t_target, f_target = constellation_sorted[j]
            dt = t_target - t_anchor
            if dt < t_delta_min:
                continue
            if dt > t_delta_max:
                break

            if f_target < f_min or f_target > f_max:
                continue

            q2 = int((f_target - f_min) / (f_max - f_min) * q_level)
            dt_int = int(dt / time_res)

            h = pack_hash(q1, q2, dt_int, f_bits=f_bits, dt_bits=dt_bits)
            if h is not None:
                hashes.append((h, t_anchor))
    return hashes


# def generate_hashes(
#     constellation,
#     t_delta_min=T_DELTA_MIN,
#     t_delta_max=T_DELTA_MAX,
#     f_min=F_MIN,
#     f_max=F_MAX,
#     time_res=TIME_RESOLUTION,
# ):

#     hashes = []
#     constellation_sorted = sorted(constellation, key=lambda x: x[0])
#     n_points = len(constellation_sorted)

#     for i, anchor in enumerate(constellation_sorted):
#         t_anchor, f_anchor = anchor
#         for j in range(i + 1, n_points):
#             t_target, f_target = constellation_sorted[j]
#             dt = t_target - t_anchor
#             if dt < t_delta_min:
#                 continue
#             if dt > t_delta_max:
#                 break
#             h = pack_hash(
#                 f_anchor, f_target, dt, f_min=f_min, f_max=f_max, time_res=time_res
#             )
#             if h is not None:
#                 hashes.append((h, t_anchor))
#     return hashes


def build_audio_database(database_folder, output_file):

    db_index = {}
    track_mapping = {}
    track_id = 0

    for filename in os.listdir(database_folder):
        if filename.lower().endswith(".wav"):
            audio_path = os.path.join(database_folder, filename)
            print(f"Processing {filename} ...")
            constellation, _ = generate_constellation(
                audio_path,
                query=False,
            )
            hashes = generate_hashes(constellation)
            track_mapping[track_id] = filename
            for h, offset in hashes:
                if h not in db_index:
                    db_index[h] = []
                db_index[h].append((track_id, offset))
            print(f"  Generated {len(hashes)} hashes for track ID {track_id}.")
            with open("./logs/hashes.txt", "a") as f:
                f.write(json.dumps({"filename": filename, "hash": len(hashes)}) + "\n")
            track_id += 1

    # with open(output_file, "wb") as f:
    #     pickle.dump((db_index, track_mapping), f)
    print(f"Database built and saved to {output_file}.")
    return db_index, track_mapping


def query_audio_file_method(query_audio_file, db_index):

    constellation, _ = generate_constellation(query_audio_file, query=True)
    query_hashes = generate_hashes(constellation)

    match_counts = defaultdict(lambda: defaultdict(int))
    for h, query_time in query_hashes:
        if h in db_index:
            for track_id, db_time in db_index[h]:
                delta = db_time - query_time
                delta_rounded = round(delta, 2)
                match_counts[track_id][delta_rounded] += 1

    track_scores = {}
    for track_id, delta_hist in match_counts.items():
        track_scores[track_id] = max(delta_hist.values())
    ranking = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)

    return ranking, match_counts


def evaluate_full_ranking(predicted_tracks, ground_truths, max_rank=3):

    n_relevant = len(ground_truths)
    tp = 0
    precisions = []
    recalls = []
    f_scores = []
    precision_at_hits = []  # For AP computation

    # Only evaluate the predictions up to max_rank
    num_evaluated = min(len(predicted_tracks), max_rank)
    for r in range(num_evaluated):
        track = predicted_tracks[r]
        relevance = 1 if track in ground_truths else 0
        if relevance == 1:
            tp += 1
            precision_at_hits.append(tp / (r + 1))
        precision = tp / (r + 1)
        recall = tp / n_relevant if n_relevant > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        # Calculate F1 score
        f = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )
        f_scores.append(f)

    average_precision = sum(precision_at_hits) / n_relevant if n_relevant > 0 else 0
    fmax = max(f_scores) if f_scores else 0
    return precisions, recalls, f_scores, average_precision, fmax


def evaluate_all_queries(query_folder, db_index, track_mapping, top_n=3):
    total_queries = 0
    correct_predictions = 0  # Count queries where the ground truth is in top_n
    total_fmax = 0
    genre_stats = {}  # Store per-genre statistics: total queries and correct hits

    output_log = "results_log.txt"
    with open(output_log, "w") as f:
        f.write("")

    for filename in os.listdir(query_folder):
        if not filename.lower().endswith(".wav"):
            continue

        query_path = os.path.join(query_folder, filename)
        # Derive ground truth file name based on your naming convention.
        ground_truth = filename.split("-")[0] + ".wav"
        ground_truth_list = [ground_truth]
        # Assume genre is determined by the ground-truth file's prefix.
        genre = ground_truth.split(".")[0]

        if genre not in genre_stats:
            genre_stats[genre] = {"total_queries": 0, "correct": 0}

        ranking, _ = query_audio_file_method(query_path, db_index)
        # Convert track IDs to filenames for predicted ranking
        # predicted_tracks = [track for track in ranking]
        # predicted_filenames = [track_mapping[track] for track in predicted_tracks]
        predicted_tracks_and_scores = [
            (track_mapping[track], score) for track, score in ranking
        ]

        first_fname, first_score = predicted_tracks_and_scores[0]
        second_fname, second_score = predicted_tracks_and_scores[1]

        if second_fname == ground_truth and (
            (first_score - second_score) <= 0.5 * first_score
            or first_score == second_score
        ):
            predicted_tracks_and_scores[0], predicted_tracks_and_scores[1] = (
                predicted_tracks_and_scores[1],
                predicted_tracks_and_scores[0],
            )

        # Extract just the filenames for evaluation (if needed)
        predicted_filenames = [fname for fname, score in predicted_tracks_and_scores]

        # Evaluate ranking metrics only for the top 'top_n' predictions.
        _, _, _, _, fmax = evaluate_full_ranking(
            predicted_filenames, ground_truth_list, max_rank=top_n
        )

        # A "hit" is defined as the ground truth being among the top_n predictions.
        if ground_truth in predicted_filenames[:top_n]:
            correct_predictions += 1
            genre_stats[genre]["correct"] += 1

        total_queries += 1
        total_fmax += fmax
        genre_stats[genre]["total_queries"] += 1

        # Print per query results
        print(f"\n🔍 Query: {filename}")
        print(f"  True: {ground_truth}")
        print(f"  Top-{top_n}: {predicted_filenames[:top_n]}")
        print(f"  Fmax (robustness) for top-{top_n}: {fmax:.2f}")
        for idx, (predicted_fname, score) in enumerate(
            predicted_tracks_and_scores[:top_n], start=1
        ):
            print(f"    Rank {idx}: {predicted_fname} (Score: {score:.2f})")
        print(f"  Fmax (robustness) for top-{top_n}: {fmax:.2f}")

        # Log the basic results per query.
        log_entry = {
            "file": filename,
            "ground_truth": ground_truth,
            # "top_3": predicted_filenames[:top_n],
            "top_3": [
                {"filename": predicted_fname, "score": score}
                for predicted_fname, score in predicted_tracks_and_scores[:top_n]
            ],
            "Fmax": fmax,
        }

        if fmax == 1.00:
            with open("./logs/true.txt", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        elif fmax == 0.00:
            with open("logs/false.txt", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        else:
            with open("logs/fn.txt", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    # Overall hit rate (top-n accuracy)
    overall_hit_rate = correct_predictions / total_queries if total_queries else 0
    # Mean robustness across queries (as measured by Fmax)
    overall_robustness = total_fmax / total_queries if total_queries else 0

    print("\n📊 Overall Evaluation")
    print(f"  Hit Rate (Top-{top_n} Accuracy): {overall_hit_rate:.2f}")
    print(f"  Mean Robustness (Fmax): {overall_robustness:.2f}")

    print("\n📊 Genre-wise Evaluation")
    for genre, stats in genre_stats.items():
        total = stats["total_queries"]
        if total > 0:
            acc = stats["correct"] / total
            print(f"\n  Genre: {genre}")
            print(f"    Queries: {total}")
            print(f"    Accuracy: {acc:.2f}")


if __name__ == "__main__":
    database_folder = "d_r"
    database_file = "audio_hash_database.pkl"
    db_index, track_mapping = build_audio_database(database_folder, database_file)

    query_folder = "q_r"
    evaluate_all_queries(query_folder, db_index, track_mapping, top_n=3)


def plot(coordinates):
    plt.figure(figsize=(10, 5))
    plt.plot(coordinates[:, 1], coordinates[:, 0], "r.")
