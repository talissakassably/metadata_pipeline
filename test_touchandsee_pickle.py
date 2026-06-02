# -*- coding: utf-8 -*-

import pickle
import traceback
import inspect
from pathlib import Path


DATASET = Path(
    r"C:\Users\tkassably\Downloads\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681"
)


def patch_old_neo_pickle_compatibility():
    """
    Compatibility patch for old Neo pickles.

    Some old Neo pickles store annotations=None.
    Newer Neo reconstruction functions expect annotations to be a dict.
    """

    try:
        import neo.core.analogsignal as analogsignal

        for function_name in [
            "_new_AnalogSignalArray",
            "_new_AnalogSignal",
        ]:
            if not hasattr(analogsignal, function_name):
                continue

            original_function = getattr(analogsignal, function_name)
            signature = inspect.signature(original_function)

            def make_patched_function(original_function, signature):
                def patched_function(*args, **kwargs):
                    bound = signature.bind_partial(*args, **kwargs)

                    if "annotations" in signature.parameters:
                        if bound.arguments.get("annotations") is None:
                            bound.arguments["annotations"] = {}

                    if "array_annotations" in signature.parameters:
                        if bound.arguments.get("array_annotations") is None:
                            bound.arguments["array_annotations"] = {}

                    return original_function(*bound.args, **bound.kwargs)

                return patched_function

            setattr(
                analogsignal,
                function_name,
                make_patched_function(original_function, signature),
            )

        print("Applied old Neo pickle compatibility patch")

    except Exception as error:
        print("Could not apply Neo compatibility patch:")
        print(repr(error))


patch_old_neo_pickle_compatibility()

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
            print(
                "First event labels preview:",
                list(ev.labels[:10]) if hasattr(ev, "labels") else None,
            )

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