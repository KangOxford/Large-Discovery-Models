"""Isolated RDKit worker for canonicalizing generated SMILES batches."""

from __future__ import annotations

import json
import sys

from rdkit import Chem, RDLogger


def main() -> int:
    RDLogger.DisableLog("rdApp.*")
    smiles = json.load(sys.stdin)
    canonical: list[str | None] = []
    for text in smiles:
        try:
            mol = Chem.MolFromSmiles(str(text))
            value = Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
            canonical.append(value if value and "." not in value else None)
        except Exception:
            canonical.append(None)
    json.dump(canonical, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
