"""
Is there anything DECODABLE in this tissue?

The Phase 0 loop failed because the decoder output was indistinguishable from a
constant. Before spending anything on closed-loop access, answer the prior
question on real data, open-loop:

    do these 22 channels carry shared, low-dimensional structure,
    or are they 22 independent Poisson sources?

If independent Poisson, there is no neural manifold, nothing to align a decoder
to, and DishBrain-style control of this substrate is not going to work no matter
how good the loop code is. That is a result worth having for the cost of zero.

Every measure is run against a matched null:
  * ISI       -> exponential (Poisson) expectation, CV = 1
  * pairwise  -> spike trains independently circularly shifted (kills
                 cross-channel timing, keeps each channel's own rate and ISI)
  * PCA       -> same shift null
  * bursts    -> Poisson process at the measured rate
"""
import numpy as np

from mealib import load, detect_with_null

X, fs, _, _ = load()
# Recomputed from the recording rather than loaded from a .npz, so this script
# runs on a fresh clone with nothing but the HDF5 file.
_, spk, rate, _, _, live = detect_with_null(X, fs, seed=0)
N = X.shape[1]; T = N/fs
ch = np.flatnonzero(live)
print(f"{len(ch)} live channels, {T:.1f} s\n")
rng = np.random.default_rng(1)

# ---------- 1. ISI: Poisson or not ----------
print("1. INTERSPIKE INTERVALS   (CV = 1 is Poisson, <1 regular, >1 bursty)")
cvs = []
for c in ch:
    if len(spk[c]) < 30: continue
    isi = np.diff(spk[c])/fs
    cv = isi.std()/isi.mean()
    cvs.append(cv)
    if rate[c] > 1.0:
        print(f"   E{c:<3} n={len(isi):5d}  mean ISI {isi.mean()*1000:7.1f} ms  CV {cv:5.2f}"
              f"  frac<10ms {np.mean(isi<0.010):.3f}")
print(f"   -> median CV across {len(cvs)} channels: {np.median(cvs):.2f}\n")

# ---------- binned population matrix ----------
BIN = 0.050
edges = np.arange(0, T+BIN, BIN)
R = np.vstack([np.histogram(spk[c]/fs, bins=edges)[0] for c in ch]).astype(float)
nb = R.shape[1]

def shift_null(R, rng):
    """Circularly shift each channel independently: rate and ISI preserved,
    cross-channel timing destroyed. The correct null for 'is there coupling'."""
    return np.vstack([np.roll(r, rng.integers(nb)) for r in R])

# ---------- 2. pairwise correlation ----------
def meanabs_offdiag(M):
    C = np.corrcoef(M); np.fill_diagonal(C, np.nan)
    return np.nanmean(np.abs(C)), np.nanmax(np.abs(C)), C

m, mx, C = meanabs_offdiag(R)
nulls = [meanabs_offdiag(shift_null(R, rng))[0] for _ in range(50)]
nm, ns = np.mean(nulls), np.std(nulls)
z = (m-nm)/ns
print(f"2. PAIRWISE CORRELATION  (50 ms bins)")
print(f"   measured mean |r| {m:.4f}   max |r| {mx:.3f}")
print(f"   shift null        {nm:.4f} +/- {ns:.4f}     z = {z:+.1f}")
iu = np.triu_indices_from(C, 1)
print(f"   strongest pairs: " + ", ".join(
    f"E{ch[i]}-E{ch[j]}:{C[i,j]:+.2f}" for i,j in
    zip(*np.unravel_index(np.argsort(np.abs(np.nan_to_num(C)), axis=None)[::-1][:6], C.shape))
    if i < j)[:200] + "\n")

# ---------- 3. dimensionality ----------
def pcdim(M, frac=0.80):
    Z = (M - M.mean(1, keepdims=True))
    sd = Z.std(1, keepdims=True); sd[sd==0]=1; Z/=sd
    ev = np.linalg.svd(Z, compute_uv=False)**2
    ev /= ev.sum()
    return int(np.searchsorted(np.cumsum(ev), frac)+1), ev

d, ev = pcdim(R)
dn = [pcdim(shift_null(R, rng))[0] for _ in range(50)]
print(f"3. DIMENSIONALITY")
print(f"   PCs for 80% variance: measured {d}  /  shift null {np.mean(dn):.1f} +/- {np.std(dn):.1f}"
      f"   (of {len(ch)} channels)")
print(f"   PC1 explains {ev[0]*100:.1f}%   PC1-3 {ev[:3].sum()*100:.1f}%\n")

# ---------- 4. population bursts ----------
pop = R.sum(0)
lam = pop.mean()
thr = lam + 4*np.sqrt(lam)
nb_meas = int((pop > thr).sum())
nb_pois = int((rng.poisson(lam, (200, nb)) > thr).sum()/200)
print(f"4. POPULATION BURSTS  (bin count > mean + 4*sqrt(mean) = {thr:.1f} spikes / 50 ms)")
print(f"   measured bins: {nb_meas} / {nb}   Poisson null: {nb_pois:.1f}"
      f"   ratio {nb_meas/max(nb_pois,1e-9):.2f}\n")

# ---------- 5. stationarity / drift ----------
half = nb//2
r1 = R[:, :half].mean(1)/BIN; r2 = R[:, half:2*half].mean(1)/BIN
print(f"5. STATIONARITY  (first 60 s vs second 60 s)")
print(f"   rate correlation across channels: r = {np.corrcoef(r1, r2)[0,1]:.3f}")
print(f"   population rate {r1.sum():.1f} Hz -> {r2.sum():.1f} Hz "
      f"({100*(r2.sum()-r1.sum())/r1.sum():+.1f}%)")

# ---------- 6. timescale sweep ----------
# Structure could live at a bin width other than 50 ms, so sweep it.
#
# NOT with dim80. An integer PC count has almost no variance under the shift
# null -- 30 surrogates often return the identical integer -- so its z is
# either 0/0 or dominated by a single quantisation step. A one-unit change
# then reads as overwhelming significance, which is the same class of error as
# reporting a hit rate without measuring chance.
#
# Participation ratio is continuous and has a usable null distribution.

def pratio(M):
    """(sum lambda)^2 / sum lambda^2. ~1 if one component dominates,
    ~n_channels if variance is spread evenly. Continuous, so it has a null."""
    Z = M - M.mean(1, keepdims=True)
    sd = Z.std(1, keepdims=True); sd[sd == 0] = 1; Z = Z / sd
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())

print(f"\n6. TIMESCALE SWEEP  (participation ratio vs shift null)")
print(f"   {'bin':>8} {'bins':>7} {'PR':>7} {'null':>14} {'z':>7} {'dim80':>7}")
print("   " + "-" * 54)
sweep = []
for b in (0.005, 0.010, 0.020, 0.050, 0.100, 0.250, 0.500, 1.000):
    e = np.arange(0, T + b, b)
    Rb = np.vstack([np.histogram(spk[c] / fs, bins=e)[0] for c in ch]).astype(float)
    nb = Rb.shape[1]
    pr = pratio(Rb)
    prn = [pratio(np.vstack([np.roll(r, rng.integers(nb)) for r in Rb]))
           for _ in range(30)]
    mu, sd = float(np.mean(prn)), float(np.std(prn))
    z = (pr - mu) / sd if sd > 1e-9 else float("nan")
    sweep.append((b, nb, pr, mu, sd, z))
    print(f"   {b*1000:6.0f}ms {nb:7d} {pr:7.2f} {mu:7.2f} +/-{sd:5.2f} "
          f"{z:+7.1f} {pcdim(Rb)[0]:7d}")

zs = [abs(s[5]) for s in sweep if s[5] == s[5]]          # drop NaN
worst = max(sweep, key=lambda s: abs(s[5]) if s[5] == s[5] else -1)
print(f"\n   largest |z| across timescales: {abs(worst[5]):.1f} "
      f"at {worst[0]*1000:.0f} ms bins")
print("   -> no bin width at which shared structure appears"
      if zs and max(zs) < 3.0 else
      "   -> structure appears at some timescale -- the 50 ms result is not "
      "the whole story")

# ---------- 7. duplicate-unit control ----------
# The PR deficit is strongest at 5-20 ms and decays as bins widen. That is the
# signature of one unit recorded on two adjacent electrodes, not of network
# coupling -- a duplicate fires at the SAME sample on both channels, while
# synaptic coupling carries a lag.
#
# So test it directly before calling anything network structure.

def sync_frac(a, b, fs, win_ms=1.0):
    """Fraction of spikes in `a` with a spike in `b` within +/- win_ms.
    High -> the two electrodes are seeing one unit."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    w = win_ms * fs / 1000.0
    idx = np.searchsorted(b, a)
    lo = np.clip(idx - 1, 0, len(b) - 1)
    hi = np.clip(idx, 0, len(b) - 1)
    d = np.minimum(np.abs(b[lo] - a), np.abs(b[hi] - a))
    return float(np.mean(d <= w))

print("\n7. DUPLICATE-UNIT CONTROL  (coincidence within 1 ms)")
BINC = 0.020
ec = np.arange(0, T + BINC, BINC)
Rc = np.vstack([np.histogram(spk[c] / fs, bins=ec)[0] for c in ch]).astype(float)
Cc = np.corrcoef(Rc); np.fill_diagonal(Cc, 0.0)

drop = set()
pairs = [(i, j) for i in range(len(ch)) for j in range(i + 1, len(ch))
         if abs(Cc[i, j]) > 0.15]
if not pairs:
    print("   no pair exceeds |r| = 0.15 at 20 ms bins")
for i, j in pairs:
    sf = sync_frac(spk[ch[i]], spk[ch[j]], fs)
    same = sf > 0.10
    print(f"   E{ch[i]:<3}-E{ch[j]:<3} r={Cc[i, j]:+.2f}  coincidence={sf:.3f}   "
          f"{'SAME UNIT' if same else 'distinct'}")
    if same:
        drop.add(i if rate[ch[i]] < rate[ch[j]] else j)

keep = [k for k in range(len(ch)) if k not in drop]
print(f"   dropping {len(drop)} channel(s) as duplicates -> {len(keep)} remain")

if drop:
    Rk = Rc[keep]
    nbk = Rk.shape[1]
    prk = pratio(Rk)
    prnk = [pratio(np.vstack([np.roll(r, rng.integers(nbk)) for r in Rk]))
            for _ in range(30)]
    muk, sdk = float(np.mean(prnk)), float(np.std(prnk))
    zk = (prk - muk) / sdk if sdk > 1e-9 else float("nan")
    print(f"   PR at 20 ms after removal: {prk:.2f}   null {muk:.2f} +/- {sdk:.2f}"
          f"   z = {zk:+.1f}")
    print("   -> the deficit was duplicate units, not network coupling"
          if abs(zk) < 3.0 else
          "   -> deficit survives duplicate removal: consistent with genuine "
          "shared structure")

np.savez("structure.npz", R=R, C=C, ev=ev, ch=ch, bin_s=BIN,
         sweep=np.array(sweep, dtype=float),
         dropped=np.array(sorted(drop), dtype=int))
