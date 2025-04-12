import os
import pickle
import time
from collections import defaultdict

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np

# from helpers import evaluate_topk_metrics
from scipy.ndimage import maximum_filter

"""
Audio file processing base parameters
"""
SAMPLE_RATE = 22500
FFT_WINDOW_SIZE = 4096
HOP_SIZE = 512


"""
Peak picking parameters
"""
PEAK_NEIGHBORHOOD_SIZE = (20, 20)
MINIMUM_DB_THRESHOLD = -40


"""
Hash generation time parameters
"""
MIN_HASH_TIME_DELTA = 0.0
MAX_HASH_TIME_DELTA = 2.0
TIME_QUANTISATION_RESOLUTION = 0.01


"""
Hash generation frequency parameters (Limiting frequency bands)
"""
MIN_FREQ = 30
MAX_FREQ = 4000


"""
Hash packing parameters
"""
FREQ_QUANTISATION_BITS = 10
TIM_QUANTISATION_BITS = 10


def generate_constellation(audio_file):
    """
    Loads the audio file (using `librosa`), computes the STFT, then identifies peaks above a
    configurable dB threshold. Returns both a `time_freq_coordinates` array and the decibel
    spectrogram for optional visualization.

    Input:
        audio_file (str): Path to the audio file.

    Output:
        time_freq_coordinates (np.ndarray): 2D array of (time, frequency) coordinates.
        extracted_decibels (np.ndarray): Decibel-scaled spectrogram.
    """
    y, SR = librosa.load(audio_file, sr=SAMPLE_RATE, mono=False)
    S = np.abs(librosa.stft(y, n_fft=FFT_WINDOW_SIZE, hop_length=HOP_SIZE))
    extracted_decibels = librosa.amplitude_to_db(S, ref=np.max)

    local_max = maximum_filter(extracted_decibels, size=PEAK_NEIGHBORHOOD_SIZE)
    peaks = (extracted_decibels == local_max) & (
        extracted_decibels > MINIMUM_DB_THRESHOLD
    )
    peak_indices = np.argwhere(peaks)

    times = librosa.frames_to_time(peak_indices[:, 1], sr=SR, hop_length=HOP_SIZE)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=FFT_WINDOW_SIZE)[peak_indices[:, 0]]

    time_freq_coordinates = np.vstack((times, freqs)).T
    return time_freq_coordinates, extracted_decibels


def pack_hash(anchor_freq, target_freq, delta_time):
    """
    Packs hash and corresponding time and ID values into a 32-bit integer.

    Input:
        anchor_freq (int): Quantized anchor frequency.
        target_freq (int): Quantized target frequency.
        delta_time (int): Quantized time difference.

    Output:
        int: A 32-bit integer hash (or None if delta_time exceeds range).
    """
    if delta_time >= (2**TIM_QUANTISATION_BITS):
        return None
    return (
        (anchor_freq << (TIM_QUANTISATION_BITS + FREQ_QUANTISATION_BITS))
        | (target_freq << TIM_QUANTISATION_BITS)
        | delta_time
    )


def generate_hashes(constellation):
    """
    Sorts the peaks by time, then pairs each anchor peak with subsequent peaks
    within a maximum time difference (`MAX_HASH_TIME_DELTA`).
    Quantizes frequencies to FREQ_QUANTISATION_BITS and time differences using
    TIME_QUANTISATION_RESOLUTION. Skips freq that lie outside the min and max frequency bounds
    Pack them.

    Input:
        constellation (np.ndarray): Array of (time, frequency) coordinates.

    Output:
        list: Sorted list of tuples (hash, anchor_time).
    """
    hashes = []
    sorted_constellation = sorted(constellation, key=lambda x: x[0])
    num_of_peaks = len(sorted_constellation)
    quantisation = (2**FREQ_QUANTISATION_BITS) - 1

    for i, anchor in enumerate(sorted_constellation):
        anchor_time, anchor_freq = anchor

        if anchor_freq < MIN_FREQ or anchor_freq > MAX_FREQ:
            continue

        anchor_freq = int(
            (anchor_freq - MIN_FREQ) / (MAX_FREQ - MIN_FREQ) * quantisation
        )

        for j in range(i + 1, num_of_peaks):
            target_time, target_freq = sorted_constellation[j]
            time_delta = target_time - anchor_time
            if time_delta < MIN_HASH_TIME_DELTA:
                continue

            if time_delta > MAX_HASH_TIME_DELTA:
                break

            if target_freq < MIN_FREQ or target_freq > MAX_FREQ:
                continue

            target_freq = int(
                (target_freq - MIN_FREQ) / (MAX_FREQ - MIN_FREQ) * quantisation
            )
            time_delta = int(time_delta / TIME_QUANTISATION_RESOLUTION)

            hash = pack_hash(anchor_freq, target_freq, time_delta)
            if hash is not None:
                hashes.append((hash, anchor_time))

    hashes.sort(key=lambda token_info: token_info[0])
    return hashes


def query_audio_file_method(query_audio_file, db_index):
    """
    Creates hashes for the query audio, then scans the database index to find matches.
    Tallies score by how often the track’s hash offsets align with the query.
    Ranks the tracks in descending order of alignment score.
    It rounds the time offset difference to two decimals so that nearly identical times
    are treated as the same “bin,”

    Input:
        query_audio_file (str): Path to the query audio file.
        db_index (dict): Inverted index mapping hash values to (track_id, time) tuples.

    Output:
        tuple: (ranking, match_counts) where ranking is a sorted list of (track_id, score)
               and match_counts is a dictionary with detailed matching counts.
    """
    constellation, _ = generate_constellation(query_audio_file)
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


def fingerprintBuilder(database_folder, db_output_file_path):
    """
    Build (or load, if available) the fingerprint database from the given database folder.

    Input:
        database_folder (str): Path to the folder containing database audio files (.wav).
        fingerprintDatabaseOutputFile (str): Path where the fingerprint database
                                             will be saved/loaded.

    Output:
        tuple: (db_index, track_mapping)
    """
    if os.path.exists(db_output_file_path):
        with open(db_output_file_path, "rb") as f:
            db_index, track_mapping = joblib.load(f)
        print(f"Loaded existing fingerprint database from {db_output_file_path}.")
    else:
        db_index, track_mapping = build_audio_database(
            database_folder, db_output_file_path
        )
    return db_index, track_mapping


def build_audio_database(database_folder, output_file):
    """
    Iterates over all .wav files in the database folder, generating constellation maps
    and, in turn, hash tokens. Aggregates results into an inverted index mapping each
    32-bit hash to (track_id, offset_time). Saves the index and track mapping for reuse.
    calls the generate constellation and generate_hashes functions to create the hashes
    for each audio file.

    Input:
        database_folder (str): Folder containing database audio files (.wav).
        output_file (str): Path to save the fingerprint database.

    Output:
        tuple: (db_index, track_mapping)
    """
    db_index = {}
    track_mapping = {}
    track_id = 0
    for filename in os.listdir(database_folder):
        if filename.lower().endswith(".wav"):
            audio_path = os.path.join(database_folder, filename)
            constellation, _ = generate_constellation(audio_path)
            hashes = generate_hashes(constellation)
            track_mapping[track_id] = filename

            for hash, offset in hashes:
                if hash not in db_index:
                    db_index[hash] = []
                db_index[hash].append((track_id, offset))
            track_id += 1
            print(f"***Processed file #{track_id}: {filename}")

    with open(output_file, "wb") as f:
        pickle.dump((db_index, track_mapping), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Database built and saved to {output_file}.")
    return db_index, track_mapping


def audioIdentification(query_folder, fingerprintDatabasePath, output_path):
    """
    Identifies query audio files by matching them against the fingerprint database and returns the top 3 ranking file names
    For each query file, writes a line in the format:
        query_audio.wav    ground_truth    match1.wav    match2.wav    match3.wav    recall_at_3

    Input:
        query_folder (str): Folder containing query audio files (.wav).
        fingerprintDatabasePath (str): Path to the fingerprint database file.
        output_path (str): Path to the output text file.
    """
    if not os.path.exists(fingerprintDatabasePath):
        raise FileNotFoundError(
            f"Fingerprint database not found at: {fingerprintDatabasePath}"
        )

    start_inference = time.time()

    with open(fingerprintDatabasePath, "rb") as f:
        db_index, track_mapping = pickle.load(f)
    print(f"Loaded fingerprint database from {fingerprintDatabasePath}.")

    output_lines = []
    for filename in os.listdir(query_folder):
        if not filename.lower().endswith(".wav"):
            continue

        query_path = os.path.join(query_folder, filename)
        ranking, _ = query_audio_file_method(query_path, db_index)
        top3 = ranking[:3]
        top3_filenames = [
            track_mapping.get(track_id, "unknown") for track_id, _ in top3
        ]

        ###***###
        """The below lines only get called if you want to the recall, you can output it in the line variable, will need ot be imported from helpers file"""
        # ground_truth = filename.split("-")[0] + ".wav"
        # ground_truth_list = [ground_truth]
        # predicted_filenames = [
        #     track_mapping.get(track_id, "unknown") for track_id, _ in ranking
        # ]
        # recall_at_3 = evaluate_topk_metrics(predicted_filenames, ground_truth_list, k=3)
        ###***###

        line = "\t".join([filename] + top3_filenames)
        output_lines.append(line)

    with open(output_path, "w") as f_out:
        for line in output_lines:
            f_out.write(line + "\n")

    end_inference = time.time()
    print(f"Inference completed in {end_inference - start_inference:.2f} seconds.")
    print(f"Audio identification results written to {output_path}.")


if __name__ == "__main__":

    """
    database_folder : Path to the folder containing database audio files (.wav).
    database_file : Path to the file that will be created to save fingerprints.
    query_folder : Path to the folder containing query audio files (.wav).
    output_file : Path to the output text file that will be created for results.

    By calling the script the code runs in the following order:
     - If the Db is built, it will load the db, this might take some time, since loading into the memory(but not too long too)
     - else it will build the db, fairly fast
     - automatically starts the identification process after building the db or using the built db
     - The output will be saved in the output_file
    """
    database_folder = ""
    database_file = "audio_fingerprint_database.pkl"
    query_folder = ""
    output_file = "output.txt"

    start_time = time.time()
    db_index, track_mapping = fingerprintBuilder(database_folder, database_file)
    end_db_build = time.time()

    print(f"Database building completed in {end_db_build - start_time:.2f} seconds.")

    audioIdentification(query_folder, database_file, output_file)

    total_time = time.time() - start_time
    print(
        f"Entire pipeline (DB build + inference) completed in {total_time:.2f} seconds."
    )
