# Environment-aware extraction guide

Some datasets do **not** belong in the same Python environment.

## Why

The legacy touchscreen data and some old pickle/Neo/NIX files were created with old Python/Neo versions. Trying to load them in a modern/base environment can fail or silently lose details.

So the pipeline is split into:

## Base environment

Use this for:

- openfield extraction
- NWB extraction
- TouchAndSee compact extraction if it works
- biological table export
- case study

Run:

```powershell
conda activate base
py run_pipeline_env_aware.py --config configs\config_template.json --steps extract_base
```

## neo_legacy environment

Use this for:

- legacy touchscreen extraction
- old Neo/NIX pickle extraction

Run:

```powershell
conda activate neo_legacy

py extractors\extract_legacy_touchscreen_metadata.py ^
  "C:\Users\tkassably\Downloads\d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-data" ^
  --read_lfp ^
  --output_folder outputs\extracted_metadata
```

## Then return to base

```powershell
conda activate base

py run_pipeline_env_aware.py --config configs\config_template.json --steps biological_tables case_study
```

## Important rule

Do not force everything through one Python environment.

The correct architecture is:

```text
base env extraction outputs
+ neo_legacy extraction outputs
→ same outputs/extracted_metadata folder
→ biological table export in base
→ case study in base
```
