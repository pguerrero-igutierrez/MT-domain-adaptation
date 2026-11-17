"""Recover the reported clinical evaluation inputs from the agreeing public caches."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_NAMES = ['Latxa-Qwen3-VL-8B-Instruct', 'generalv1', 'clinicalv1', 'clinicalv2']


def recover_test_set(paths):
    expected = None
    for path in paths:
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'{path} is not a non-empty prediction list.')
        samples = []
        for row in rows:
            if row.get('direction') != 'ca2eu' or not all(isinstance(row.get(k), str) for k in ('source', 'reference')):
                raise ValueError(f'{path} contains an invalid clinical evaluation record.')
            samples.append({'source': row['source'], 'target': row['reference'], 'direction': row['direction']})
        if expected is not None and expected != samples:
            raise ValueError(f'{path} disagrees with the other recorded evaluation inputs.')
        expected = samples
    if expected is None:
        raise ValueError('At least one prediction cache is required.')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/test_set_clinical_reported.json')
    args = parser.parse_args()
    paths = [ROOT / 'outputs/eval' / f'{name}_clinical_ca2eu_backup.json' for name in CACHE_NAMES]
    samples = recover_test_set(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Recovered {len(samples)} inputs confirmed by all four clinical caches -> {args.output}')


if __name__ == '__main__':
    main()
