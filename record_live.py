"""
Record from FinalSpark's public LiveMEA stream. No account, no key, no cost.

The official FinalSpark-np/LiveMEA client is DEAD: it points at
livemeaservice.finalspark.com, which now returns 502, and its documented
event names ("livedata", "meaid") no longer exist. The service moved to
livenp.finalspark.com with a different protocol. This file implements the
current one, recovered from the live client bundle:

    GET  /controllers                 -> controller id, MEA index, electrode layout
    GET  /sample                      -> {"numbersample": 1600} samples per packet
    socket.io connect (same origin)
    emit "start" {id_intan, mea_index, channel_index:[...]}
    on   "data"  -> flat list of 32*1600 float32 microvolts

    fs = 3750 Hz  (Intan 30 kHz / 8), 1600 samples = 0.4267 s per packet

Usage:
    python record_live.py --seconds 60 --out live_data.h5
"""
from __future__ import annotations
import argparse, asyncio, json, time, urllib.request
from datetime import datetime, timezone

import numpy as np
import h5py
import socketio

BASE = "https://livenp.finalspark.com"
FS = 3750.0


def get_controller():
    ctrl = json.load(urllib.request.urlopen(BASE + "/controllers"))["controllers"][0]
    nsamp = json.load(urllib.request.urlopen(BASE + "/sample"))["numbersample"]
    return ctrl, nsamp


def reshape_packet(flat, n_ch, n_samp):
    """Decide channel-major vs sample-major by smoothness.

    A neural trace is continuous in time, so the correct reshape has far
    smaller sample-to-sample differences than the wrong one. Interleaved data
    read as channel-major looks like white noise; this measures that.
    """
    a = np.asarray(flat, dtype=np.float32)
    chan_major = a.reshape(n_ch, n_samp)
    samp_major = a.reshape(n_samp, n_ch).T
    r1 = np.abs(np.diff(chan_major, axis=1)).mean()
    r2 = np.abs(np.diff(samp_major, axis=1)).mean()
    return (chan_major, "channel_major") if r1 <= r2 else (samp_major, "sample_major")


async def record(seconds: float, out: str):
    ctrl, nsamp = get_controller()
    ch = ctrl["default_electrodes"]
    payload = {"id_intan": ctrl["_id"], "mea_index": ctrl["default_mea"],
               "channel_index": ch}
    print(f"controller {ctrl['name']} ({ctrl['_id']})  online={ctrl['online']}  "
          f"MEA {ctrl['default_mea']}  {len(ch)} electrodes  {nsamp} samples/packet")

    need = int(np.ceil(seconds * FS / nsamp))
    packets, stamps, order = [], [], [None]

    sio = socketio.AsyncClient()

    @sio.event
    async def connect():
        await sio.emit("start", payload)

    @sio.on("data")
    async def on_data(flat):
        arr, how = reshape_packet(flat, len(ch), nsamp)
        order[0] = how
        packets.append(arr)
        stamps.append(datetime.now(timezone.utc).isoformat())
        print(f"  {len(packets)}/{need} packets", end="\r")

    @sio.on("maintenance")
    async def on_maint(d):
        print(f"\n  MAINTENANCE: {d}")

    await sio.connect(BASE, transports=["websocket"], wait_timeout=30)
    t0 = time.time()
    while len(packets) < need and time.time() - t0 < seconds * 4 + 60:
        await sio.sleep(0.2)
    await sio.disconnect()

    data = np.concatenate(packets, axis=1)          # (32, N)
    print(f"\ncaptured {data.shape[1]} samples x {data.shape[0]} ch "
          f"= {data.shape[1]/FS:.1f} s   layout={order[0]}")

    with h5py.File(out, "w") as f:
        f.create_dataset("data_uV", data=data, compression="gzip")
        f.create_dataset("packet_times_utc",
                         data=np.array(stamps, dtype="S32"))
        f.attrs.update(fs_hz=FS, n_channels=data.shape[0],
                       samples_per_packet=nsamp, layout=order[0],
                       source=BASE, controller=ctrl["_id"],
                       mea_index=ctrl["default_mea"],
                       recorded_utc=datetime.now(timezone.utc).isoformat())
        f.create_dataset("electrode_layout",
                         data=np.array(json.dumps(ctrl["electrode_layout"]),
                                       dtype=h5py.string_dtype()))
    print(f"wrote {out}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="live_data.h5")
    a = ap.parse_args()
    asyncio.run(record(a.seconds, a.out))
