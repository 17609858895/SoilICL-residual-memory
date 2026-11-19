"""Build the VisNIR-SOC modelling table (OSSL v1.2), amendment 4.

Common window 400-2500 nm (1,051 points; LUCAS spectra start at 400 nm).
Reflectance -> absorbance log10(1/R) with R clipped at 1e-4. Exact duplicate spectra removed (first kept).
Outputs: data/visnir_soc_X.npy (float32, absorbance, not yet SNV), data/visnir_soc_meta.parquet
Coordinates stay in the private meta file and are used only for spatial blocks and aggregated maps.
"""
import gzip, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data"
KEY = ["dataset.code_ascii_txt", "id.layer_uuid_txt"]
EXPECTED_SHA = "e5a08c0deda182b7c123505c9f39adf9f5c9f6313c0022e74c51bc1f10b27a83"

h = hashlib.sha256()
with open(D / "ossl_visnir_L0_v1.2.csv.gz", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert h.hexdigest() == EXPECTED_SHA, "VisNIR file hash mismatch"

with gzip.open(D / "ossl_visnir_L0_v1.2.csv.gz", "rt") as f:
    header = f.readline().strip().split(",")
spec_all = [c for c in header if c.startswith("scan_visnir.")]
spec_cols = [c for c in spec_all if 400 <= int(c.split(".")[1].split("_")[0]) <= 2500]
assert len(spec_cols) == 1051, len(spec_cols)
dtypes = {c: "float32" for c in spec_cols}

parts_X, parts_k = [], []
for ch in pd.read_csv(D / "ossl_visnir_L0_v1.2.csv.gz", usecols=KEY + ["scan.visnir.model.name_utf8_txt"] + spec_cols,
                      dtype=dtypes, chunksize=5000, na_values=["", "NA"], low_memory=False):
    ch = ch[ch[spec_cols].notna().all(axis=1)]
    parts_X.append(ch[spec_cols].to_numpy(np.float32))
    parts_k.append(ch[KEY + ["scan.visnir.model.name_utf8_txt"]].reset_index(drop=True))
R = np.vstack(parts_X)
keys = pd.concat(parts_k, ignore_index=True)
print("VisNIR rows with complete 400-2500 nm spectra:", len(keys))

lab_cols = ["oc_usda.c729_w.pct", "c.tot_usda.a622_w.pct", "caco3_usda.a54_w.pct", "n.tot_usda.a623_w.pct"]
lab = pd.read_csv(D / "ossl_soillab_L1_v1.2.csv.gz", usecols=KEY + lab_cols, na_values=["", "NA"], low_memory=False)
site_cols = ["longitude.point_wgs84_dd", "latitude.point_wgs84_dd", "id.dataset.site_ascii_txt",
             "layer.upper.depth_usda_cm", "layer.lower.depth_usda_cm", "location.country_iso.3166_txt"]
site = pd.read_csv(D / "ossl_soilsite_L0_v1.2.csv.gz", usecols=KEY + site_cols, na_values=["", "NA"], low_memory=False)
meta = keys.merge(lab, on=KEY, how="left", validate="one_to_one").merge(site, on=KEY, how="left", validate="one_to_one")
mask = meta["oc_usda.c729_w.pct"].notna().to_numpy()
R, meta = R[mask], meta[mask].reset_index(drop=True)
print("VisNIR rows with SOC:", len(meta))

# exact duplicate spectra: keep the first occurrence
hashes = pd.util.hash_pandas_object(pd.DataFrame(R), index=False).to_numpy()
dup = pd.Series(hashes).duplicated().to_numpy()
print("exact duplicate spectra removed:", int(dup.sum()))
R, meta = R[~dup], meta[~dup].reset_index(drop=True)

A = np.log10(1.0 / np.clip(R, 1e-4, None)).astype(np.float32)
meta = meta.rename(columns={"dataset.code_ascii_txt": "source", "oc_usda.c729_w.pct": "soc",
                            "c.tot_usda.a622_w.pct": "tc", "caco3_usda.a54_w.pct": "caco3",
                            "n.tot_usda.a623_w.pct": "tn", "longitude.point_wgs84_dd": "lon",
                            "latitude.point_wgs84_dd": "lat", "id.dataset.site_ascii_txt": "site_id"})
cell = np.where(meta["lon"].notna() & meta["lat"].notna(),
                "cell_" + np.floor(meta["lon"].fillna(0)).astype(int).astype(str) + "_" +
                np.floor(meta["lat"].fillna(0)).astype(int).astype(str), None)
site_key = np.where(meta["site_id"].notna(), "site_" + meta["source"] + "_" + meta["site_id"].astype(str), None)
meta["group_block"] = pd.Series(cell).fillna(pd.Series(site_key)).fillna("src_" + meta["source"]).values

np.save(D / "visnir_soc_X.npy", A)
meta.to_parquet(D / "visnir_soc_meta.parquet", index=False)
print("final n:", len(meta)); print(meta["source"].value_counts().to_string())
print("n spatial groups:", meta["group_block"].nunique())
print("rows with point coordinates:", int(meta["lon"].notna().sum()))
