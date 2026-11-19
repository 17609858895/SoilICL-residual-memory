"""Shared split logic for the leave-source-out (P2) and fold-as-target (P1) runs.
See PILOT_PROTOCOL.md, amendments 3 and 4."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

SEED = 42
# data key -> (spectra file, meta file, CNN bin width, results root)
DATASETS = {"mir": ("mir_soc_X.npy", "mir_soc_meta.parquet", 8, "P2"),
            "visnir": ("visnir_soc_X.npy", "visnir_soc_meta.parquet", 4, "V_visnir")}
SCENARIOS = {
    "mir": {"AfSIS": ["AFSIS1.SSL", "AFSIS2.SSL"], "ICRAF": ["ICRAF.ISRIC"], "LUCAS_Woodwell": ["LUCAS.WOODWELL.SSL"],
            "SCHIEDUNG": ["SCHIEDUNG.SSL"], "SERBIA": ["SERBIA.SSL"], "KSSL": ["KSSL.SSL"]},
    "visnir": {"Europe": ["LUCAS.SSL", "LUCAS.WOODWELL.SSL"], "ICRAF": ["ICRAF.ISRIC"], "KSSL": ["KSSL.SSL"]},
}
ADAPT_GROUP_COL = {
    "mir": {"AfSIS": "group_block", "ICRAF": "group_block", "LUCAS_Woodwell": "group_block",
            "SCHIEDUNG": "site_key", "KSSL": "group_block"},
    "visnir": {"Europe": "group_block", "ICRAF": "group_block", "KSSL": "group_block"},
}


def get_split(meta: pd.DataFrame, scenario: str, data: str = "mir"):
    """Return train idx, target idx, adaptation pool idx, local test idx (pool/test None if no adaptation).
    scenario 'foldK' = spatial-block fold K of a 5-fold GroupKFold (P1); otherwise a held-out source set (P2)."""
    if scenario.startswith("fold"):
        k = int(scenario[4:])
        fold = np.full(len(meta), -1)
        for j, (_, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                                    .split(np.zeros(len(meta)), groups=meta["group_block"].values)):
            fold[te] = j
        return np.where(fold != k)[0], np.where(fold == k)[0], None, None
    tgt_mask = meta["source"].isin(SCENARIOS[data][scenario]).to_numpy()
    tr, tg = np.where(~tgt_mask)[0], np.where(tgt_mask)[0]
    pool = test = None
    col = ADAPT_GROUP_COL[data].get(scenario)
    if col is not None:
        if col == "site_key":
            groups = (meta["source"] + "_" + meta["site_id"].astype(str)).to_numpy()[tg]
        else:
            groups = meta[col].to_numpy()[tg]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(tg, groups=groups))
        pool, test = tg[a], tg[b]
    return tr, tg, pool, test


def bootstrap_groups(meta: pd.DataFrame, rows, scenario: str, data: str = "mir"):
    """Grouping used for bootstrap CIs within a target set."""
    if scenario.startswith("fold") or ADAPT_GROUP_COL[data].get(scenario) == "group_block":
        return meta["group_block"].values[rows]
    return (meta["source"] + "_" + meta["site_id"].astype(str)).values[rows]
