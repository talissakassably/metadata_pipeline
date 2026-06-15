# CA1 meta-analysis pipeline — AFC/ACM improved

Replace your current `meta_analysis` files with these files.

Main fixes:
- AFC/ACM no longer uses `source_type`, `source_format`, `recording_system`, or raw `behavioral_context` by default.
- This prevents the categorical analysis from simply rediscovering dataset/file-format identity.
- AFC/ACM now removes constant and extremely rare one-hot categories.
- AFC/ACM plot uses jitter and dataset centroids so overlapping profiles are visible.
- A new categorical drivers figure is produced: `figures/07_categorical_afc_mca_top_drivers.png`.

Run from repository root:

```powershell
py meta_analysis\run_ca1_metadata_meta_analysis.py `
  outputs\extracted_metadata\d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json `
  outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json `
  outputs\extracted_metadata\legacy_touchscreen_metadata.json `
  outputs\extracted_metadata\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json `
  --output_dir meta_analysis\outputs
```

If your filenames differ, run:

```powershell
dir outputs\extracted_metadata
```

Then replace the names in the command.
