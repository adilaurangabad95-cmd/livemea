"""
First look at real data -- with a measured null, because Phase 0.

Spike detection on extracellular MEA data is threshold crossing on a bandpassed
trace. Noise crosses thresholds too. So every number here is reported against a
phase-randomised surrogate: same length, same power spectrum, same amplitude
distribution, NO temporal structure. Anything the surrogate reproduces is not a
finding.
"""
import numpy as np
from scipy.signal import welch

from mealib import load, detect_with_null

X, fs, lay, _ = load()
nch, N = X.shape

# Bandpass, detect, then detect again on a phase-randomised surrogate.
# The surrogate is the whole point of this script -- see mealib.surrogate.
B, spk, rate, srate, thr, live = detect_with_null(X, fs, seed=0)
print()

print(f"{'ch':>3} {'rms_uV':>8} {'thr_uV':>8} {'rate_Hz':>9} {'null_Hz':>9} {'ratio':>7}")
print("-"*50)
for c in range(nch):
    r = rate[c]/srate[c] if srate[c] > 0 else np.inf
    flag = "  <-- real" if (rate[c] > 0.05 and r > 2.0) else ""
    print(f"{c:3d} {B[c].std():8.2f} {thr[c]:8.2f} {rate[c]:9.3f} {srate[c]:9.3f} {r:7.2f}{flag}")

print(f"\nTOTAL  measured {rate.sum():.2f} Hz   surrogate null {srate.sum():.2f} Hz"
      f"   ratio {rate.sum()/max(srate.sum(),1e-9):.2f}")
print(f"electrodes with real spiking: {live.sum()} / {nch}")

# population burst structure: only meaningful if the above passed
bins = 20e-3
edges = np.arange(0, N/fs + bins, bins)
pop = np.zeros(len(edges)-1)
for c in np.flatnonzero(live):
    pop += np.histogram(spk[c]/fs, bins=edges)[0]
if live.sum():
    print(f"\npopulation rate: mean {pop.mean()/bins:.2f} Hz  max {pop.max()/bins:.2f} Hz")
    print(f"Fano factor (var/mean of 20 ms counts): {pop.var()/max(pop.mean(),1e-9):.2f}"
          "   (1.0 = Poisson, >1 = bursting)")
f, P = welch(X, fs=fs, nperseg=4096, axis=1)
band = lambda lo, hi: P[:, (f>=lo)&(f<hi)].mean()
print(f"\npower: 1-10Hz {band(1,10):.1f}  10-100Hz {band(10,100):.2f}  "
      f"300-1400Hz {band(300,1400):.4f} uV^2/Hz")
np.save("spikes.npy", np.array(spk, dtype=object), allow_pickle=True)
np.savez("summary.npz", rate=rate, srate=srate, thr=thr, live=live, pop=pop)
