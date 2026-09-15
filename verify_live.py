"""Is this live tissue, or a looping file?

The controller metadata lists recorded_file="wavelet_signal.bin", which is
exactly what a replay fixture looks like. Two tests:
  1. internal: does the 120 s recording repeat itself at any lag?
  2. external: does a fresh grab match the old one anywhere?
A replay shows a near-1.0 normalised correlation peak. Live tissue does not.
"""
import asyncio, json, time, urllib.request
import numpy as np, socketio

from mealib import load

BASE = "https://livenp.finalspark.com"
X, FS, _, _ = load()
c = int(np.argmax(X.std(axis=1)))
old = X[c].astype(np.float64); old -= old.mean()

# --- 1. internal loop test ---
n = len(old)
ac = np.fft.irfft(np.abs(np.fft.rfft(old, 2*n))**2, 2*n)[:n]
ac /= ac[0]
lags = np.arange(n)/FS
m = lags > 1.0
print(f"internal autocorrelation, ch{c}: max |r| beyond 1 s lag = {np.abs(ac[m]).max():.4f} "
      f"at {lags[m][np.argmax(np.abs(ac[m]))]:.2f} s   (replay would be ~1.0)")

# --- 2. fresh grab vs stored ---
ctrl = json.load(urllib.request.urlopen(BASE+"/controllers"))["controllers"][0]
pay = {"id_intan": ctrl["_id"], "mea_index": ctrl["default_mea"],
       "channel_index": ctrl["default_electrodes"]}
pk = []
async def grab():
    sio = socketio.AsyncClient()
    @sio.event
    async def connect(): await sio.emit("start", pay)
    @sio.on("data")
    async def d(f): pk.append(np.asarray(f, dtype=np.float32).reshape(32,1600))
    await sio.connect(BASE, transports=["websocket"], wait_timeout=30)
    t0=time.time()
    while len(pk) < 40 and time.time()-t0 < 120: await sio.sleep(0.2)
    await sio.disconnect()
asyncio.run(grab())
new = np.concatenate(pk, axis=1)[c].astype(np.float64); new -= new.mean()
print(f"fresh grab: {new.size/FS:.1f} s")

xc = np.fft.irfft(np.fft.rfft(old, 2*n) * np.conj(np.fft.rfft(new, 2*n)), 2*n)
xc /= np.sqrt((old**2).sum()*(new**2).sum())
print(f"cross-correlation old vs fresh: max |r| = {np.abs(xc).max():.4f}   (replay would be ~1.0)")
print("\nVERDICT:", "REPLAY / CANNED FILE" if np.abs(xc).max() > 0.5
      else "independent samples -- consistent with a live stream")
