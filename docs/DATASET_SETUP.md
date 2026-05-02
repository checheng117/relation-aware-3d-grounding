# Dataset setup (bring your own data)

Use Conda environment **`rag3d`** (Python **3.10**); create it via `make env` or `bash scripts/setup_env.sh` as described in the root **README.md**.

## Environment variables

1. Copy `cp .env.example .env`
2. Set `HF_TOKEN` only if you need Hugging Face gated models or datasets.
3. Never commit `.env` or embed tokens in YAML or logs.

## Layout (`raw_root`, default `data/raw/referit3d`)

The repository does not ship benchmarks or download URLs. Arrange local data as:

```
<raw_root>/
  scans/
    scene0000_00/
      scene0000_00_vh_clean_aggregation.json   # ScanNet-style aggregation (segGroups + OBB when present)
    scene0001_00/
      ...
  annotations/
    train.csv
    val.csv
    test.csv
```

Alternatively, use a **single** CSV with a `split` column (`train` / `val` / `test`) and set `combined_csv: utterances.csv` in `configs/dataset/referit3d.yaml`.

### Minimum CSV columns (case-insensitive)

| Column | Description |
|--------|-------------|
| `scene_id` | Matches folder name under `scans/` (e.g. `scene0000_00`) |
| `utterance` | Referring expression |
| `target_object_id` | Matches `objectId` in the aggregation JSON |

Optional: `utterance_id`, `relation_type_gold` (or `relation`), `split` (for combined CSV).

### Processed output (`processed_dir`, default `data/processed`)

`prepare_data.py` writes:

- `train_manifest.jsonl`, `val_manifest.jsonl`, `test_manifest.jsonl`
- `dataset_summary.json`

For fast iteration **without** ScanNet:

```bash
python scripts/prepare_data.py --mode mock-debug --config configs/dataset/referit3d.yaml
```

writes under `data/processed/debug/` (size controlled by `debug.mock` in the dataset config).

### Importing existing JSONL

If you already have manifests, set `import_jsonl: path/to/file.jsonl` in the dataset config, then:

```bash
python scripts/prepare_data.py --mode build --config configs/dataset/referit3d.yaml
```

### Common commands

```bash
python scripts/prepare_data.py --mode validate --config configs/dataset/referit3d.yaml
python scripts/prepare_data.py --mode build --config configs/dataset/referit3d.yaml
python scripts/prepare_data.py --mode template --template-out data/processed/template_manifest.json
```

Geometry-backed NR3D-style builds may use `--mode build-nr3d-geom` when your config enables that path; ensure aggregation JSONs exist under each `scans/<scene_id>/` tree.

### Official data pointers

- [ReferIt3D](https://github.com/referit3d/referit3d)
- [ScanNet](http://www.scan-net.org/)

After download, point `raw_root` at your merged tree so `scans/` and `annotations/` match the layout above.
