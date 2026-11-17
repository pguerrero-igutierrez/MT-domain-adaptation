"""Run COMET in its own interpreter, without importing it into the MT environment."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace


class ExternalCometEvaluator:
    def __init__(self, python, checkpoint='Unbabel/wmt22-comet-da'):
        self.python = str(python)
        self.checkpoint = checkpoint

    def predict(self, data, batch_size=8, gpus=0):
        with tempfile.TemporaryDirectory(prefix='mt-comet-') as temporary:
            temporary = Path(temporary)
            source, target = temporary / 'input.json', temporary / 'scores.json'
            source.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            command = [self.python, str(Path(__file__).with_name('score_comet.py')),
                       '--input', str(source), '--output', str(target),
                       '--checkpoint', self.checkpoint, '--batch-size', str(batch_size),
                       '--gpus', str(gpus)]
            subprocess.run(command, check=True)
            result = json.loads(target.read_text(encoding='utf-8'))
        return SimpleNamespace(system_score=result['system_score'])


def validate_cached_predictions(saved, samples, direction):
    """A cache is reusable only for the same ordered inputs and references."""
    if len(saved) != len(samples):
        raise ValueError('Cached predictions have a different sample count; use a fresh --output-dir.')
    for index, (prediction, sample) in enumerate(zip(saved, samples)):
        if (prediction.get('source') != sample['source']
                or prediction.get('reference') != sample['target']
                or prediction.get('direction') != direction
                or not isinstance(prediction.get('hypothesis'), str)):
            raise ValueError(f'Cached prediction {index} does not match this test set; use a fresh --output-dir.')
