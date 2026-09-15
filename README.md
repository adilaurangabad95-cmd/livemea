# livemea — a working client for FinalSpark's public neuron stream

FinalSpark run living human brain organoids on microelectrode arrays in Vevey,
Switzerland, and stream one array publicly — no account, no API key, no cost.

**The official client no longer works.** This one does, plus spike detection
against a matched null and a dimensionality analysis of the result.

```bash
git clone https://github.com/<you>/livemea && cd livemea
pip install -r requirements.txt
python analyse.py          # runs against the committed 120 s recording
```

A recording is included, so everything here reproduces without touching the
network.

---

## Why this exists

`FinalSpark-np/LiveMEA` on GitHub points at `livemeaservice.finalspark.com`,
which returns **502** on every endpoint, and emits socket events (`livedata`,
`meaid`) the live service no longer implements. Anyone following that README
gets nothing and no useful error.

The service moved to `livenp.finalspark.com` with a different protocol.
Recovered by reading the live client bundle:

```
GET  https://livenp.finalspark.com/controllers
        -> controller _id, default_mea, default_electrodes, electrode_layout
GET  https://livenp.finalspark.com/sample
        -> {"numbersample": 1600}

socket.io  (same origin, transports=["websocket"])
  emit "start"  {"id_intan": <_id>, "mea_index": 0, "channel_index": [0..31]}
  on   "data"   -> flat list of 32*1600 float32, CHANNEL-MAJOR, microvolts
  on   "maintenance"

fs = 3750 Hz   (Intan 30 kHz / 8)
1600 samples = 0.4267 s per packet, arriving every ~0.36 s
```

Packet layout — channel-major versus sample-major — is documented nowhere, so
`record_live.py` decides per packet by comparing sample-to-sample smoothness of
both reshapes, and stores the verdict in the file. Interleaved data read the
wrong way looks like white noise; this measures that rather than guessing.

## Files

| File | What |
|---|---|
| `mealib.py` | loading and spike detection, shared by everything below |
| `record_live.py` | the recorder → HDF5 |
| `verify_live.py` | proves the stream is live, not a replayed file |
| `analyse.py` | spike detection against a phase-randomised surrogate null |
| `structure.py` | is the population activity low-dimensional? |
| `plot_live.py` | the five-panel figure |
| `live_data_int16.h5` | 120 s, int16 @ 0.1 µV/LSB (0.5% of signal SD) |
| `livemea_first_look.png` | the figure |

```python
from mealib import load
X, fs, layout, attrs = load()     # X in microvolts, (32, 451200); fs = 3750.0
```

`load()` accepts either schema — `data_uV` float microvolts, or `data` int16
with a `gain_uV_per_LSB` attribute — and returns microvolts either way.

## Is the stream actually live?

The controller advertises `recorded_file: "wavelet_signal.bin"`, which is
exactly what a replay fixture looks like. So it was tested, not assumed:

- **internal loop test** — autocorrelation of the 120 s trace at every lag > 1 s:
  peak |r| = **0.0125**. A looping file would show ~1.0.
- **independent-grab test** — a fresh 17 s capture cross-correlated against the
  stored recording: peak |r| = **0.0097**.

Two independent samples of the same tissue, not the same bytes twice.

## Spike detection, against a null

4th-order Butterworth 300–1400 Hz, threshold at 4.5 × median(|x|)/0.6745
(Quiroga), 1 ms refractory.

**The null** is the same detector run on a phase-randomised surrogate —
identical power spectrum, identical amplitude distribution, no temporal
structure. Anything the surrogate reproduces is a threshold artefact, not a
spike. Reporting MEA spike rates without this is reporting the noise floor.

| | |
|---|---|
| Population rate, measured | **69.9 Hz** |
| Population rate, surrogate null | **0.42 Hz** |
| Ratio | **165×** |
| Electrodes above null | **22 / 32** |
| Busiest electrode | E9, **15.7 Hz** |
| Noise RMS (bandpassed) | 4.5–15.6 µV |
| Fano factor, 20 ms bins | **1.12** |

**Spatial structure is real.** MEA 0 is four 8-electrode wells ~3 mm apart.
Well 1 carries the activity (E9 15.7, E12 8.4, E15 5.9 Hz); well 3 is silent
(E27, E30, E31 all 0.00 Hz). Activity is localised to tissue, not spread
uniformly as an electrical artefact would be.

**Fano 1.12 is the surprise.** Neuronal cultures normally show heavy
synchronised bursting (Fano ≫ 1). This one is close to Poisson. Either these
neurospheres are genuinely desynchronised, or the 0.427 s packet boundaries are
clipping bursts. Worth checking before any conclusion rests on it.

## Is there anything decodable? (`structure.py`)

The prior question, asked open-loop: do these 22 channels carry shared,
low-dimensional structure, or are they 22 independent Poisson sources?

Null for every measure: each spike train circularly shifted by an independent
random lag. Rate and ISI preserved, cross-channel timing destroyed.

| Measure | Measured | Null | Verdict |
|---|---|---|---|
| Median ISI CV | **0.91** | 1.0 = Poisson | ~Poisson, not bursty |
| Mean pairwise \|r\| (50 ms) | **0.0216** | 0.0163 ± 0.0009 | z = +5.7, but tiny |
| PCs for 80% variance | 17 of 22 | 17.4 ± 0.5 | **statistic too coarse — see below** |
| PC1 variance explained | **6.2%** | 5.3% | ≈ chance (1/22 = 4.5%) |
| Population bursts > μ+4σ | 11 / 2407 bins | 2.0 | 5.5×, but n = 11 |
| Rate stability, 1st vs 2nd minute | **r = 0.997** | — | very stable |

### Why the PC count is the wrong statistic

An earlier version of this analysis concluded "no shared structure" because
dim80 (17) sat on its shift null (17.4 ± 0.5). **That conclusion was an artefact
of the statistic.** An integer PC count has almost no variance under the null —
30 surrogates frequently return the identical integer — so its z is either 0/0
or dominated by one quantisation step. It cannot resolve a sub-1% effect.

Participation ratio, (Σλ)²/Σλ², is continuous and has a usable null. Swept
across bin widths:

| bin | PR | shift null | z | dim80 |
|---|---|---|---|---|
| 5 ms | 21.67 | 21.98 ± 0.00 | −128.0 | 17 |
| 20 ms | 21.52 | 21.92 ± 0.01 | −44.6 | 17 |
| 50 ms | 21.37 | 21.81 ± 0.02 | −19.4 | 17 |
| 250 ms | 20.44 | 21.08 ± 0.06 | −10.6 | 16 |
| 1000 ms | 18.31 | 18.71 ± 0.24 | −1.7 | 14 |

**Shared structure is present at every timescale.** Note that the z gradient is
driven by bin count, not by a timescale-specific mechanism — the PR *deficit*
itself is roughly flat to slightly increasing with bin width (0.31 → 0.81),
while z falls simply because there are fewer bins to estimate from. Reading the
z column as "fine-timescale coupling" would be a sample-size artefact.

### The duplicate-unit control

Two electrode pairs carry the highest correlations, and both are physically
adjacent — the signature of one unit recorded twice. Measured rather than
assumed, as coincidence within ±1 ms:

| pair | r (20 ms) | coincidence ±1 ms | verdict |
|---|---|---|---|
| E10–E11 | +0.28 | **0.263** | same unit |
| E16–E18 | +0.29 | **0.340** | same unit |

Dropping the lower-rate member of each pair leaves 20 channels. The deficit
**survives**: PR 19.84 against a null of 19.94 ± 0.01, z = −13.6 at 20 ms.
Duplicates accounted for most of the magnitude, not for the existence.

### Conclusion

**There is genuine shared structure in this tissue's spontaneous activity, and
it is minute.** After removing duplicate units, the population's effective
dimensionality is 0.10 below independence out of 20 — about **0.5%** of the
variance structure. Statistically overwhelming (z = −13.6); substantively
almost nothing.

That is not a low-dimensional manifold. Twenty channels still require about
twenty dimensions. So there is no latent space here to align a decoder onto, and
manifold-based BCI approaches that assume one do not apply to this substrate.
But "near-independent Poisson sources" is too strong — the correct statement is
that the departure from independence is real and around half a percent.

### Three caveats, and the first one is the whole point

1. **This is spontaneous activity.** Driven activity can be far more structured
   than resting activity, and structuring it is exactly what closed-loop
   stimulation is for. **This result does not say the tissue cannot be
   organised — it says it is barely organised on its own.** Testing that
   requires stimulation, and the public stream is read-only.
2. 22 electrodes across a neurosphere is very sparse sampling; a manifold
   confined to unobserved cells would be invisible.
3. Spike sorting was not done — these are threshold crossings, not units. The
   duplicate-unit control above handles the two obvious cases; it is not a
   substitute for sorting.

## Data source and citation

The recording was captured from FinalSpark's public LiveMEA stream, which they
make freely available. **FinalSpark are the source of the biological data and
should be cited as such** in anything derived from it:

- Jordan, F. D. et al. *Open and remotely accessible Neuroplatform for research
  in wetware computing.* Frontiers in Artificial Intelligence (2024).
  https://doi.org/10.3389/frai.2024.1376042
- https://finalspark.com

The code here is independent work, not affiliated with or endorsed by
FinalSpark.

## License

Code: MIT, see `LICENSE`. The included recording is FinalSpark's biological
data, captured from their public stream and redistributed for reproducibility
with the attribution above.

## Author

Dr Adil Ahmed Ameen — Junior Resident, Department of Physiology,
All India Institute of Medical Sciences, Kalyani.
