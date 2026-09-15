"""Shared loading and spike detection for LiveMEA recordings.

Every script in this repo goes through this module, for one reason: the
recordings exist in two schemas and nothing downstream should have to know
which one it was handed.

    live_data_int16.h5   dataset "data",    int16 + attrs["gain_uV_per_LSB"]
    live_data.h5         dataset "data_uV", float32 microvolts

`load()` returns microvolts either way.

Spike detection lives here too, rather than inside one analysis script, so that
`structure.py` and `plot_live.py` can recompute spikes from the recording
instead of depending on .npz intermediates that are not in version control.
That is what makes `git clone && python analyse.py` actually work.
"""

from __future__ import annotations

import json
import os

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

# Tried in order when no path is given. The int16 file is the committed one.
CANDIDATES = ("live_data_int16.h5", "live_data.h5")

# Spike-detection defaults. Changing these changes every reported number, so
# they are named constants rather than call-site literals.
BAND_HZ = (300.0, 1400.0)
BAND_ORDER = 4
THRESHOLD_K = 4.5          # Quiroga: k * median(|x|) / 0.6745
REFRACTORY_MS = 1.0


def load(path: str | None = None):
    """Load a recording as microvolts.

    Returns
    -------
    X : (n_channels, n_samples) float32, microvolts
    fs : float, sampling rate in Hz
    layout : list of electrode dicts, or None if the file carries none
    attrs : dict of the file's HDF5 attributes
    """
    if path is None:
        for candidate in CANDIDATES:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No recording found. Looked for {CANDIDATES} in {os.getcwd()}.\n"
                f"Record one with:  python record_live.py --seconds 120"
            )

    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)

        if "data_uV" in f:
            X = f["data_uV"][:].astype("float32")
        elif "data" in f:
            gain = float(attrs.get("gain_uV_per_LSB", 1.0))
            X = f["data"][:].astype("float32") * gain
        else:
            raise KeyError(
                f"{path} has neither 'data_uV' nor 'data'. "
                f"Datasets present: {list(f)}"
            )

        if "fs_hz" not in attrs:
            raise KeyError(f"{path} has no 'fs_hz' attribute")
        fs = float(attrs["fs_hz"])

        layout = None
        if "electrode_layout" in f:
            raw = f["electrode_layout"][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            layout = json.loads(raw)

    print(f"loaded {path}: {X.shape[0]} ch x {X.shape[1]} samples "
          f"@ {fs:g} Hz = {X.shape[1] / fs:.1f} s")
    return X, fs, layout, attrs


def bandpass(X: np.ndarray, fs: float,
             band=BAND_HZ, order: int = BAND_ORDER) -> np.ndarray:
    """Zero-phase bandpass. sosfiltfilt, so no group delay on spike times."""
    sos = butter(order, list(band), btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=1)


def surrogate(B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomised surrogate: identical power spectrum and amplitude
    distribution, destroyed waveform shape and timing.

    This is the null for spike detection. Noise crosses thresholds too, so any
    rate the surrogate reproduces is a threshold artefact and not a spike.
    """
    F = np.fft.rfft(B, axis=1)
    phase = rng.uniform(0, 2 * np.pi, F.shape)
    phase[:, 0] = 0.0                       # keep the DC term real
    return np.fft.irfft(np.abs(F) * np.exp(1j * phase), n=B.shape[1], axis=1)


def detect(B: np.ndarray, fs: float, k: float = THRESHOLD_K,
           refr_ms: float = REFRACTORY_MS):
    """Threshold-crossing detection on a bandpassed trace.

    Negative-going peaks past k * median(|x|) / 0.6745, with a refractory
    period. These are threshold crossings, not sorted units.

    Returns (spike_index_arrays, rates_hz, thresholds_uV).
    """
    thr = k * np.median(np.abs(B), axis=1) / 0.6745
    refr = max(1, int(round(refr_ms * fs / 1000.0)))
    duration_s = B.shape[1] / fs

    spikes, rates = [], []
    for c in range(B.shape[0]):
        crossings = np.flatnonzero(B[c] < -thr[c])
        kept, last = [], -refr
        for i in crossings:
            if i - last >= refr:
                kept.append(i)
                last = i
        spikes.append(np.asarray(kept, dtype=np.int64))
        rates.append(len(kept) / duration_s)

    return spikes, np.asarray(rates), thr


def detect_with_null(X: np.ndarray, fs: float, seed: int = 0):
    """Full pipeline: bandpass, detect, and detect again on the surrogate.

    Returns (B, spikes, rates, surrogate_rates, thresholds, live_mask) where
    `live_mask` marks channels whose rate exceeds both 0.05 Hz and twice their
    own surrogate rate.
    """
    B = bandpass(X, fs)
    spikes, rates, thr = detect(B, fs)
    rng = np.random.default_rng(seed)
    _, srates, _ = detect(surrogate(B, rng), fs)
    live = (rates > 0.05) & (rates > 2.0 * srates)
    return B, spikes, rates, srates, thr, live
