"""Summarize SIH-SUS confirmation scores on folds shared by all finalists."""

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


FINALISTS = (
    'q19_full_auto_axes_s20',
    'q26_plaquettes_auto_axes_s20',
    'q26_full_auto_axes_s10',
    'q26_native_auto_balanced_s20',
    'q26_plaquettes_auto_shared_s20',
)


def confirmation_report(rows):
    by_fold = defaultdict(dict)
    for row in rows:
        fold = int(row['fold'])
        profile = row['name'].split('_', 2)[2]
        if fold >= 2 and profile in FINALISTS:
            if profile in by_fold[fold]:
                raise ValueError(f'Duplicate result for fold {fold} and {profile}')
            by_fold[fold][profile] = row

    matched = {fold: cases for fold, cases in sorted(by_fold.items())
               if all(profile in cases for profile in FINALISTS)}
    for fold, cases in matched.items():
        if len({float(row['classic']) for row in cases.values()}) != 1:
            raise ValueError(f'Classical baseline differs within fold {fold}')
    completed = sum(map(len, by_fold.values()))
    output = [f'**{completed}/70 attempts complete**. {len(matched)}/14 folds have all five results.']
    if not matched:
        return '\n'.join(output) + '\n'

    output += [
        '',
        '| Circuit | Mean hybrid AP | Hybrid Δ classical | Hybrid wins | Mean soft MoE AP | Soft Δ classical | Soft wins | Soft Δ hybrid |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for profile in FINALISTS:
        cases = [fold[profile] for fold in matched.values()]
        hybrid, soft, classical = ([float(row[metric]) for row in cases]
                                   for metric in ('hybrid', 'soft_moe', 'classic'))
        differences = [[value - base for value, base in zip(scores, classical)] for scores in (hybrid, soft)]
        ih, im = (sum(delta > 0 for delta in diffs) for diffs in differences)
        output.append(f'| `{profile}` | {mean(hybrid):.6f} | {mean(differences[0]):+.6f} | {ih} | '
                      f'{mean(soft):.6f} | {mean(differences[1]):+.6f} | {im} | {mean(soft)-mean(hybrid):+.6f} |')
    output += [
        '',
        '| Metric | First circuit | Second circuit | Mean paired AP gap | First wins | Second wins |',
        '|---|---|---|---:|---:|---:|',
    ]
    for metric, profiles in (('hybrid', FINALISTS[:3]), ('soft_moe', FINALISTS[3:])):
        for first, second in combinations(profiles, 2):
            gaps = [float(fold[first][metric]) - float(fold[second][metric]) for fold in matched.values()]
            output.append(f'| {metric} | `{first}` | `{second}` | {mean(gaps):+.6f} | '
                          f'{sum(gap > 0 for gap in gaps)} | {sum(gap < 0 for gap in gaps)} |')
    output += ['', 'The tables use matched confirmation folds only. Repeated folds share admissions across repeats.']
    return '\n'.join(output) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    args = parser.parse_args()
    with args.scores.open(newline='') as stream:
        print(confirmation_report(csv.DictReader(stream)), end='')
