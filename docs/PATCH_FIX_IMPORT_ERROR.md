# Patch note

Fixed extractor import error:

`extract_metadata_pipeline.py` expected `make_json_safe` and `unique_values`
from `extractors/utils.py`, but the packaged `utils.py` did not contain those
functions.

This update replaces `extractors/utils.py` with a robust shared utility file
containing:

- `make_json_safe`
- `unique_values`
- `dataframe_unique_values`
- `dataframe_column_summary`
- `safe_file_size_mb`
- `safe_relpath`

Run again:

```powershell
py run_pipeline.py --config configs\config_template.json --steps extract biological_tables case_study
```

Since openfield already finished, you can also run only the remaining parts by
running extraction again; it will overwrite/recreate the JSONs.
