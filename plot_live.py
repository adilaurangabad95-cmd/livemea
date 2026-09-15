import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mealib import load, detect_with_null

DATA, NULLC, CRIT = "#2B4ACB", "#98A3AD", "#AE2856"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})

X, fs, lay, _ = load()
B, spk, rate, srate, thr, live = detect_with_null(X, fs, seed=0)
N = X.shape[1]

# Panel B needs electrode coordinates. The int16 recording may not carry them,
# so the figure drops to two columns rather than crashing.
HAVE_LAYOUT = lay is not None and len(lay) >= X.shape[0]
if not HAVE_LAYOUT:
    print("no electrode_layout in the file -- skipping panel B (electrode map)")

fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.1, .95], hspace=.42, wspace=.28)

# A: raw traces
ax = fig.add_subplot(gs[0, :2])
best = np.argsort(rate)[::-1][:5]
t0 = int(20*fs); w = int(2*fs); t = np.arange(w)/fs + 20
for i, c in enumerate(best):
    ax.plot(t, B[c, t0:t0+w] - i*140, lw=.4, color=DATA)
    s = spk[c][(spk[c] >= t0) & (spk[c] < t0+w)]
    ax.plot(s/fs, B[c, s] - i*140, ".", ms=3.5, color=CRIT)
    ax.text(19.92, -i*140, f"E{c}", ha="right", va="center", fontsize=7, color="#555")
ax.set(xlabel="time (s)", yticks=[], title="A  bandpassed 300–1400 Hz, five busiest electrodes (red = detected spike)")
ax.set_xlim(19.9, 22)

# B: electrode map
ax = fig.add_subplot(gs[0, 2])
if HAVE_LAYOUT:
    xs = np.array([e["position"]["x"] for e in lay[:32]]); ys = np.array([e["position"]["y"] for e in lay[:32]])
    ci = np.array([e["channel_index"] for e in lay[:32]])
    r = rate[ci]
    # MEA 0 is four spatially separate 8-electrode clusters ~3 mm apart --
    # four wells, not one field. Plot cluster index vs within-cluster position
    # so the structure is visible; true µm spacing is in the HDF5 attrs.
    grp = np.rint(xs/3000.0).astype(int)
    lx = xs - grp*3000
    sc = ax.scatter(lx, ys + grp*900, s=40+300*r/max(r.max(),1e-9), c=r,
                    cmap="YlGnBu", edgecolors="#333", linewidths=.4, vmin=0)
    for k in range(grp.max()+1):
        ax.text(-260, 200 + k*900, f"well {k}", fontsize=7, color="#555", va="center")
    for cc, xx, yy, gg in zip(ci, lx, ys, grp):
        ax.annotate(str(cc), (xx, yy+gg*900), fontsize=5, ha="center", va="center")
    plt.colorbar(sc, ax=ax, label="spikes/s", shrink=.85)
    ax.set(title="B  MEA 0 — four 8-electrode wells", xlabel="µm within well",
           ylabel="well (offset)", yticks=[])
    ax.set_xlim(-420, 900)
else:
    ax.text(0.5, 0.5, "electrode layout not stored\nin this recording",
            ha="center", va="center", fontsize=8, color="#6B7883",
            transform=ax.transAxes)
    ax.set(title="B  electrode map — unavailable", xticks=[], yticks=[])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

# C: raster
ax = fig.add_subplot(gs[1, :])
for row, c in enumerate(np.flatnonzero(live)):
    ax.plot(spk[c]/fs, np.full(len(spk[c]), row), "|", ms=2.6, color=DATA, mew=.55)
ax.set(xlim=(0, N/fs), ylim=(-1, live.sum()), xlabel="time (s)",
       ylabel="electrode (rate-ordered)",
       title=f"C  raster — {live.sum()} of 32 electrodes with activity above the surrogate null")

# D: rate vs null
ax = fig.add_subplot(gs[2, :2])
i = np.arange(32); wdt = .4
ax.bar(i-wdt/2, rate, wdt, color=DATA, label="measured")
ax.bar(i+wdt/2, srate, wdt, color=NULLC, label="phase-randomised surrogate (null)")
ax.set_yscale("symlog", linthresh=.01)
ax.set(xlabel="electrode", ylabel="spikes/s (symlog)", xticks=i[::2],
       title="D  every rate against its own null — the panel Phase 0 was missing")
ax.legend(frameon=False, fontsize=7)

# E: population rate
ax = fig.add_subplot(gs[2, 2])
b = .05; ed = np.arange(0, N/fs+b, b); pop = np.zeros(len(ed)-1)
for c in np.flatnonzero(live): pop += np.histogram(spk[c]/fs, bins=ed)[0]
ax.plot(ed[:-1], pop/b, lw=.6, color=DATA)
ax.axhline(pop.mean()/b, color=CRIT, lw=.9, ls="--", label=f"mean {pop.mean()/b:.0f} Hz")
ax.set(xlabel="time (s)", ylabel="population rate (Hz)",
       title=f"E  Fano = {pop.var()/pop.mean():.2f}")
ax.legend(frameon=False, fontsize=7)

fig.suptitle("FinalSpark LiveMEA — 120 s from a living human neurosphere, Switzerland · "
             "32 electrodes @ 3750 Hz · recorded 14 Sep 2026 · cost ₹0",
             fontsize=10.5, y=.985)
fig.savefig("livemea_first_look.png", bbox_inches="tight")
print("wrote livemea_first_look.png")
