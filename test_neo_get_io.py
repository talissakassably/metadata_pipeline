# -*- coding: utf-8 -*-

import os
import traceback

print("SCRIPT STARTED")

try:
    import neo
    from neo.io import get_io
    print("Neo version:", neo.__version__)
except Exception as error:
    print("Could not import Neo")
    print(repr(error))
    traceback.print_exc()
    raise SystemExit


DATASET_PATH = r"C:\Users\tkassably\Downloads\d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-data"

print("Dataset path:")
print(DATASET_PATH)
print("Dataset exists:", os.path.exists(DATASET_PATH))

files = []

for root, dirs, filenames in os.walk(DATASET_PATH):
    for filename in filenames:
        if filename.lower().endswith((".pkl", ".nio")):
            files.append(os.path.join(root, filename))

files = sorted(files)

print("Number of .pkl/.nio files found:", len(files))

if len(files) == 0:
    print("No files found. Check the dataset path.")
    raise SystemExit

print("First five files:")
for file_path in files[:5]:
    print(file_path)

print("\nTesting first .pkl file if available...")

pkl_files = [f for f in files if f.lower().endswith(".pkl")]
nio_files = [f for f in files if f.lower().endswith(".nio")]

files_to_test = []

if len(pkl_files) > 0:
    files_to_test.append(pkl_files[0])

if len(nio_files) > 0:
    files_to_test.append(nio_files[0])

for file_path in files_to_test:
    print("\n" + "=" * 80)
    print("Testing:")
    print(file_path)
    print("=" * 80)

    try:
        io = get_io(file_path)
        print("get_io worked")
        print("IO object:", io)
        print("IO class:", io.__class__.__name__)

        try:
            data = io.read(lazy=True)
            print("io.read(lazy=True) worked")
            print("Returned:", type(data))
        except Exception as error:
            print("io.read(lazy=True) failed")
            print(repr(error))
            traceback.print_exc()

        try:
            data = io.read(lazy=False)
            print("io.read(lazy=False) worked")
            print("Returned:", type(data))
        except Exception as error:
            print("io.read(lazy=False) failed")
            print(repr(error))
            traceback.print_exc()

    except Exception as error:
        print("get_io failed")
        print(repr(error))
        traceback.print_exc()

print("\nSCRIPT FINISHED")