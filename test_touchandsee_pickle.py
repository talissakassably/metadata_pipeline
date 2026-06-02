# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
import traceback

DATASET = Path(
    r"C:\Users\tkassably\Downloads\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681"
)

files = sorted(DATASET.rglob("*.pkl"))

print("Pickle files found:", len(files))

if len(files) == 0:
    raise SystemExit("No .pkl files found")

test_file = files[0]

print("Testing file:")
print(test_file)

try:
    with open(test_file, "rb") as f:
        obj = pickle.load(f)

    print("Loaded successfully")
    print("Object type:", type(obj))

    print("Block name:", getattr(obj, "name", None))
    print("Block description:", getattr(obj, "description", None))
    print("Annotations:", getattr(obj, "annotations", None))

    segments = getattr(obj, "segments", [])
    print("Segments:", len(segments))

    if len(segments) > 0:
        seg = segments[0]
        print("First segment type:", type(seg))
        print("First segment annotations:", getattr(seg, "annotations", None))
        print("Spike trains:", len(getattr(seg, "spiketrains", [])))
        print("Analog signals:", len(getattr(seg, "analogsignals", [])))
        print("Events:", len(getattr(seg, "events", [])))
        print("Epochs:", len(getattr(seg, "epochs", [])))

        if len(getattr(seg, "spiketrains", [])) > 0:
            st = seg.spiketrains[0]
            print("First spiketrain spikes:", len(st))
            print("First spiketrain annotations:", getattr(st, "annotations", None))

        if len(getattr(seg, "analogsignals", [])) > 0:
            sig = seg.analogsignals[0]
            print("First analog signal shape:", getattr(sig, "shape", None))
            print("First analog signal units:", getattr(sig, "units", None))
            print("First analog signal sampling rate:", getattr(sig, "sampling_rate", None))

        if len(getattr(seg, "events", [])) > 0:
            ev = seg.events[0]
            print("First event name:", getattr(ev, "name", None))
            print("First event length:", len(ev))
            print("First event labels preview:", list(ev.labels[:10]) if hasattr(ev, "labels") else None)

    if hasattr(obj, "list_units"):
        print("Block list_units:", len(obj.list_units))
        if len(obj.list_units) > 0:
            unit = obj.list_units[0]
            print("First unit annotations:", getattr(unit, "annotations", None))
            print("First unit spiketrains:", len(getattr(unit, "spiketrains", [])))

except Exception as error:
    print("FAILED")
    print(repr(error))
    traceback.print_exc()