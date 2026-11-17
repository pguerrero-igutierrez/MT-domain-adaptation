"""Score saved translation triples in the dedicated COMET environment.

Accepts evaluation backups, full result JSONs with a `per_sample` list,
or COMET triples with `src`, `mt`, `ref`. Writes scores to a new JSON file.
"""
import argparse
import json
from pathlib import Path


def load_triples(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(rows, dict):
        rows = rows.get('per_sample')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must contain a non-empty list of translation triples.')
    triples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Row {index} is not an object.')
        if all(key in row for key in ('src', 'mt', 'ref')):
            triple = {key: row[key] for key in ('src', 'mt', 'ref')}
        elif all(key in row for key in ('source', 'hypothesis', 'reference')):
            triple = dict(src=row['source'], mt=row['hypothesis'], ref=row['reference'])
        else:
            raise ValueError(f'Row {index} lacks source, translation or reference fields.')
        if not all(isinstance(value, str) for value in triple.values()):
            raise ValueError(f'Row {index} contains a non-string text field.')
        triples.append(triple)
    return triples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', default='Unbabel/wmt22-comet-da')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpus', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Use a separate output path so input predictions are preserved.')
    data = load_triples(args.input)
    from comet import download_model, load_from_checkpoint
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        checkpoint = Path(download_model(args.checkpoint))
    model = load_from_checkpoint(str(checkpoint))
    model.eval()
    scores = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'checkpoint': args.checkpoint, 'n_samples': len(data),
        'system_score': float(scores.system_score),
        'COMET': round(float(scores.system_score) * 100, 2),
        'segment_scores': [float(score) for score in scores.scores],
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
