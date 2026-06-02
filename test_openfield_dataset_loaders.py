# -*- coding: utf-8 -*-

import sys
import traceback
from pathlib import Path

import numpy as np

# Compatibility patch for old dataset scripts with NumPy 2.x
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

DATASET_ROOT = Path(
    r"C:\Users\tkassably\Downloads\d-40faae41-7e72-4c3c-9abf-91dea149158d"
)

sys.path.insert(0, str(DATASET_ROOT))


from scripts import axona
from scripts import neuralynx
from scripts import mclust
from scripts import opticsv


def find_first_file(pattern):
    files = sorted(DATASET_ROOT.rglob(pattern))
    return files[0] if files else None


def test_axona():
    print("\n" + "=" * 80)
    print("Testing Axona loader")
    print("=" * 80)

    set_file = find_first_file("*.set")

    if set_file is None:
        print("No .set file found; no Axona session tested.")
        return

    basepath = set_file.with_suffix("")
    print("Axona set file:", set_file)
    print("Axona basepath:", basepath)

    try:
        recording = axona.AxonaRecording(basepath)

        print("Recording time:", recording.recording_time)
        print("Recording duration:", recording.recording_duration)
        print("Tetrodes:", recording.tetrode_list)

        try:
            print("Spike channel count:", len(recording.spikes))
            first_key = sorted(recording.spikes.keys())[0]
            print("First spike channel:", first_key)
            print("First spike n_spikes:", recording.spikes[first_key].n_spikes)
            print("First spike n_channels:", recording.spikes[first_key].n_channels)
        except Exception as error:
            print("Could not read Axona spikes:", repr(error))

        try:
            if recording.position is not None:
                print("Position sample rate:", recording.position.sample_rate)
                print("Position timestamps:", len(recording.position.timestamps))
                print("Position coordinates shape:", recording.position.coordinates.shape)
            else:
                print("No Axona position file loaded.")
        except Exception as error:
            print("Could not read Axona position:", repr(error))

        try:
            print("EEG channel count:", len(recording.eeg))
            if len(recording.eeg) > 0:
                first_eeg = sorted(recording.eeg.keys())[0]
                print("First EEG channel:", first_eeg)
                print("First EEG sample rate:", recording.eeg[first_eeg].sample_rate)
                print("First EEG signal shape:", recording.eeg[first_eeg].signal.shape)
        except Exception as error:
            print("Could not read Axona EEG:", repr(error))

    except Exception as error:
        print("Axona loader failed:")
        print(repr(error))
        traceback.print_exc()


def test_neuralynx():
    print("\n" + "=" * 80)
    print("Testing Neuralynx loader")
    print("=" * 80)

    ntt_file = find_first_file("*.ntt")

    if ntt_file is None:
        print("No .ntt file found; no Neuralynx session tested.")
        return

    session_folder = ntt_file.parent
    print("Neuralynx session folder:", session_folder)

    try:
        recording = neuralynx.NeuralynxSession(session_folder)

        print("Recording name:", recording.recording_name)
        print("Recording time:", recording.recording_time)
        print("Recording duration:", recording.recording_duration)
        print("Tetrodes:", recording.tetrode_list)

        try:
            print("Spike tetrode count:", len(recording.spikes))
            first_key = sorted(recording.spikes.keys())[0]
            print("First tetrode:", first_key)
            print("First tetrode n spike events:", len(recording.spikes[first_key]["data"]))
        except Exception as error:
            print("Could not read Neuralynx spikes:", repr(error))

        try:
            print("EEG/LFP channel count:", len(recording.eeg))
            if recording.eeg:
                first_eeg = sorted(recording.eeg.keys())[0]
                print("First EEG/LFP channel:", first_eeg)
                print("First EEG/LFP n records:", len(recording.eeg[first_eeg]["data"]))
        except Exception as error:
            print("Could not read Neuralynx EEG/LFP:", repr(error))

        try:
            print("Position records:", len(recording.position[1]))
        except Exception as error:
            print("Could not read Neuralynx position:", repr(error))

        try:
            cuts = recording.cuts
            print("MClust cuts loaded:", bool(cuts))
            if cuts:
                print("MClust cut tetrodes:", sorted(cuts.keys())[:10])
        except Exception as error:
            print("Could not read Neuralynx MClust cuts:", repr(error))

    except Exception as error:
        print("Neuralynx loader failed:")
        print(repr(error))
        traceback.print_exc()


def test_mclust():
    print("\n" + "=" * 80)
    print("Testing MClust loader")
    print("=" * 80)

    t_file = find_first_file("*.t64") or find_first_file("*.t32") or find_first_file("*.t")

    if t_file is None:
        print("No .t/.t32/.t64 file found; no MClust test.")
        return

    session_folder = t_file.parent
    print("MClust folder:", session_folder)

    try:
        cuts = mclust.load_all_cuts(session_folder)
        print("Cuts loaded:", bool(cuts))

        if cuts:
            print("Tetrodes with units:", sorted(cuts.keys()))
            first_tetrode = sorted(cuts.keys())[0]
            print("Units on first tetrode:", sorted(cuts[first_tetrode].keys()))
            first_unit = sorted(cuts[first_tetrode].keys())[0]
            header, timestamps = cuts[first_tetrode][first_unit]
            print("First unit header keys:", list(header.keys())[:10])
            print("First unit n timestamps:", len(timestamps))

    except Exception as error:
        print("MClust loader failed:")
        print(repr(error))
        traceback.print_exc()


def test_optitrack():
    print("\n" + "=" * 80)
    print("Testing OptiTrack CSV loader")
    print("=" * 80)

    csv_files = sorted(DATASET_ROOT.rglob("*.csv"))

    if len(csv_files) == 0:
        print("No CSV file found; no OptiTrack test.")
        return

    # Try the first CSV that looks like a session-date CSV.
    csv_file = csv_files[0]
    print("CSV file:", csv_file)

    try:
        opti = opticsv.OptiCSV(csv_file)

        print("Tracking name:", opti.tracking_name)
        print("Recording time:", opti.recording_time)
        print("Sample rate:", opti.sample_rate)
        print("Meta keys:", list(opti.meta.keys())[:15])

        print("Rigid bodies:", list(opti.data["rigid_bodies"].keys()))
        print("Number of timestamps:", len(opti.data["t"]))

    except Exception as error:
        print("OptiTrack loader failed:")
        print(repr(error))
        traceback.print_exc()


if __name__ == "__main__":
    print("Dataset root:", DATASET_ROOT)
    print("Dataset exists:", DATASET_ROOT.exists())

    test_axona()
    test_neuralynx()
    test_mclust()
    test_optitrack()