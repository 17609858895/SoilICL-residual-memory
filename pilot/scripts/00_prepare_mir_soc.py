"""Build the MIR-SOC modelling table for the SoilICL pilot (OSSL v1.2).

Inputs : data/ossl_mir_L0_v1.2.csv.gz, data/ossl_soillab_L1_v1.2.csv.gz,
         data/ossl_soilsite_L0_v1.2.csv.gz  (unmodified official files)
Outputs: data/mir_soc_X.npy   float32 (n, 1701) MIR absorbance 600-4000 cm-1
         data/mir_soc_meta.parquet  keys, SOC/TC/CaCO3/TN, source, grouping keys
Coordinates stay in the private meta file and are used only to build spatial
blocks; they are never exported to results.
"""
import gzip, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data"
KEY = ["dataset.code_ascii_txt", "id.layer_uuid_txt"]
EXPECTED_SHA = "8bfed513e9b565664c67b308fa253d3c401d977bef5b75529dc85ffbef2a4118"

h = hashlib.sha256()
with open(D / "ossl_mir_L0_v1.2.csv.gz", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert h.hexdigest() == EXPECTED_SHA, "MIR file hash mismatch"

with gzip.open(D / "ossl_mir_L0_v1.2.csv.gz", "rt") as f:
    header = f.readline().strip().split(",")
spec_cols = [c for c in header if c.startswith("scan_mir.")]
assert len(spec_cols) == 1701, len(spec_cols)
usecols = KEY + ["scan.mir.model.name_utf8_txt"] + spec_cols
dtypes = {c: "float32" for c in spec_cols}

parts_X, parts_k = [], []
for ch in pd.read_csv(D / "ossl_mir_L0_v1.2.csv.gz", usecols=usecols, dtype=dtypes,
                      chunksize=5000, na_values=["", "NA"], low_memory=False):
    ch = ch[ch[spec_cols].notna().all(axis=1)]
    parts_X.append(ch[spec_cols].to_numpy(np.float32))
    parts_k.append(ch[KEY + ["scan.mir.model.name_utf8_txt"]].reset_index(drop=True))
X = np.vstack(parts_X)
keys = pd.concat(parts_k, ignore_index=True)
print("MIR rows with complete spectra:", len(keys))

lab_cols = ["oc_usda.c729_w.pct", "c.tot_usda.a622_w.pct", "caco3_usda.a54_w.pct", "n.tot_usda.a623_w.pct"]
lab = pd.read_csv(D / "ossl_soillab_L1_v1.2.csv.gz", usecols=KEY + lab_cols, na_values=["", "NA"], low_memory=False)
site_cols = ["longitude.point_wgs84_dd", "latitude.point_wgs84_dd", "id.dataset.site_ascii_txt",
             "layer.upper.depth_usda_cm", "layer.lower.depth_usda_cm", "location.country_iso.3166_txt"]
site = pd.read_csv(D / "ossl_soilsite_L0_v1.2.csv.gz", usecols=KEY + site_cols, na_values=["", "NA"], low_memory=False)

meta = keys.merge(lab, on=KEY, how="left", validate="one_to_one").merge(site, on=KEY, how="left", validate="one_to_one")
assert len(meta) == len(keys)
mask = meta["oc_usda.c729_w.pct"].notna().to_numpy()
X, meta = X[mask], meta[mask].reset_index(drop=True)
meta = meta.rename(columns={"dataset.code_ascii_txt": "source", "oc_usda.c729_w.pct": "soc",
                            "c.tot_usda.a622_w.pct": "tc", "caco3_usda.a54_w.pct": "caco3",
                            "n.tot_usda.a623_w.pct": "tn", "longitude.point_wgs84_dd": "lon",
                            "latitude.point_wgs84_dd": "lat", "id.dataset.site_ascii_txt": "site_id"})

# Spatial-block group: 1 degree cell where point coordinates exist; otherwise
# dataset + site id (profile/site level); AFSIS2 (no coordinates, no site id) = one group.
cell = np.where(meta["lon"].notna() & meta["lat"].notna(),
                "cell_" + np.floor(meta["lon"].fillna(0)).astype(int).astype(str) + "_" +
                np.floor(meta["lat"].fillna(0)).astype(int).astype(str), None)
site_key = np.where(meta["site_id"].notna(), "site_" + meta["source"] + "_" + meta["site_id"].astype(str), None)
grp = pd.Series(cell).fillna(pd.Series(site_key)).fillna("src_" + meta["source"])
meta["group_block"] = grp.values

np.save(D / "mir_soc_X.npy", X)
meta.to_parquet(D / "mir_soc_meta.parquet", index=False)
print("MIR rows with SOC:", len(meta))
print(meta["source"].value_counts().to_string())
print("n spatial groups:", meta["group_block"].nunique())
