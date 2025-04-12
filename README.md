Audio fingerprinting is the process of extracting a signature from an audio signal so that it
can be efficiently matched against a large database of known signatures. This work presents
an attempt at the Shazam-inspired algorithm for audio fingerprinting, focusing on robust
frequency peak extraction and combinatorial hashing.
The fundamental principle behind Shazam-style fingerprinting is that the “peak constel-
lation map” derived from the time-frequency representation of a track remains stable under
various forms of audio degradation, including additive noise, compression artifacts, and slight
speed fluctuations. Once these peaks are detected, a combinatorial hashing strategy trans-
forms local peak pairs into concise integer tokens (hashes). Efficient indexing of these tokens
allows real-time or near-real-time identification from an audio snippet
