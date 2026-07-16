#!/usr/bin/env python3
"""Plot all existing methods plus the new Reservoir LDM parallel baseline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ldm-parallel-root', type=Path, default=Path('outputs/ldm_reservoir_nchoices_5antigen_5seed_200eval'))
    p.add_argument('--output', type=Path, default=Path('outputs/comparison/plots/all_methods_with_ldm_parallel.png'))
    p.add_argument('--title', default='All methods comparison (5 antigens)')
    args = p.parse_args()

    methods = [
        Path('outputs/ldm/ldm_ninit20_iter200'),
        args.ldm_parallel_root,
        Path('outputs/llm_baseline/llm_baseline_5x5_200'),
        Path('outputs/reproduction/BO_transformed_overlap_optim_res.csv'),
        Path('outputs/reproduction/HEBO_optim_res.csv'),
        Path('outputs/reproduction/TURBO_optim_res.csv'),
        Path('outputs/reproduction/BO_COMBO_optim_res.csv'),
        Path('outputs/reproduction/RS_optim_res.csv'),
    ]
    labels = [
        'LDM (5 seeds)',
        'LDM parallel (5 seeds)',
        'Pure LLM (5 seeds)',
        'AntBO (10 seeds)',
        'HEBO (10 seeds)',
        'TURBO (10 seeds)',
        'COMBO (10 seeds)',
        'RS (10 seeds)',
    ]
    colors = [
        '#e41a1c',
        '#000000',
        '#0000cc',
        '#2ca02c',
        '#17becf',
        '#ff9900',
        '#984ea3',
        '#7f7f7f',
    ]
    missing = [str(path) for path in methods if not path.exists()]
    if missing:
        raise SystemExit('Missing method paths:\n' + '\n'.join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'scripts/plot_comparison.py',
        '--methods', ','.join(str(p) for p in methods),
        '--labels', ','.join(labels),
        '--colors', ','.join(colors),
        '--antigens', 'test_5_antigens.txt',
        '--output', str(args.output),
        '--title', args.title,
        '--n-column', '3',
        '--font-scale', '0.72',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
