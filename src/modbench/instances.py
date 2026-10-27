"""Versioned instance serialization (JSON metadata + NPZ arrays)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
import scipy.sparse as sp

from .core import Instance

SCHEMA_VERSION = "1.0"


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def save(inst: Instance, base_path: str | Path,
         best_known_value: Optional[float] = None,
         optimality_status: str = "unknown") -> None:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    adj = inst.adj.tocsr()
    planted = np.asarray(inst.planted if inst.planted is not None else np.array([], dtype=np.int64),
                         dtype=np.int64)
    np.savez(base_path.with_suffix(".npz"),
             weights=inst.weights,
             adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
             adj_shape=np.array(adj.shape, dtype=np.int64),
             planted=planted)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "n": int(inst.n),
        "k": int(inst.k),
        "generator_meta": inst.meta,
        "planted": planted.tolist(),
        "best_known_value": best_known_value,
        "optimality_status": optimality_status,
        "arrays_sha256": {
            "weights": _sha(inst.weights),
            "adj_data": _sha(adj.data),
            "planted": _sha(planted),
        },
    }
    base_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def load(base_path: str | Path) -> Instance:
    base_path = Path(base_path)
    meta = json.loads(base_path.with_suffix(".json").read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema mismatch: {meta.get('schema_version')} != {SCHEMA_VERSION}")
    with np.load(base_path.with_suffix(".npz")) as z:
        weights = z["weights"]
        adj = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]),
                            shape=tuple(int(v) for v in z["adj_shape"]))
        planted = z["planted"]
    return Instance(n=int(meta["n"]), k=int(meta["k"]),
                    weights=weights, adj=adj,
                    planted=planted if planted.size else None,
                    meta=meta["generator_meta"])
