"""Regression checks for data handoffs, cache reuse and isolated COMET scoring."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from comet_bridge import ExternalCometEvaluator, validate_cached_predictions
from score_comet import load_triples
from prepare_reported_clinical_test import recover_test_set


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowTests(unittest.TestCase):
    def test_reported_test_set_requires_agreement_across_prediction_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first.json', Path(directory) / 'second.json'
            rows = [{'source': 'Input', 'reference': 'Reference', 'hypothesis': 'Prediction', 'direction': 'ca2eu'}]
            for path in (first, second):
                path.write_text(json.dumps(rows))
            self.assertEqual(recover_test_set([first, second]), [{'source': 'Input', 'target': 'Reference', 'direction': 'ca2eu'}])
            second.write_text(json.dumps([dict(rows[0], reference='Different reference')]))
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                recover_test_set([first, second])

    def test_aligned_sample_is_accepted_by_translation_step(self):
        sample = load_script('03_sample_eu_literary.py')
        translate = load_script('06_translate_eu_literary.py')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sample.ES_FILE = directory / 'sample.es'
            sample.EU_FILE = directory / 'sample.eu'
            sample.OUTPUT_DIR = directory / 'sampled-data'
            sample.OUTPUT_FILE = sample.OUTPUT_DIR / 'ehuhac_sampled_parallel.jsonl'
            sample.ES_FILE.write_text('Una frase española.\nOtra frase alineada.\n')
            sample.EU_FILE.write_text('Euskarazko esaldi bat.\nBeste esaldi bat.\n')
            sample.main()
            groups = translate.load_sampled(sample.OUTPUT_FILE)
            self.assertEqual(len(groups['EhuHac']), 2)
            self.assertEqual(groups['EhuHac'][0]['source_es'], 'Una frase española.')

    def test_monolingual_legacy_sample_is_rejected_before_generation(self):
        translate = load_script('06_translate_eu_literary.py')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'legacy.jsonl'
            source.write_text(json.dumps({'doc_id': 'x', 'para_id': 0, 'source_eu': 'Test text'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'source_es'):
                translate.load_sampled(source)

    def test_cache_requires_same_order_sources_and_references(self):
        samples = [{'source': 'one', 'target': 'bat'}, {'source': 'two', 'target': 'bi'}]
        saved = [{'source': 'one', 'reference': 'bat', 'hypothesis': 'bat', 'direction': 'ca2eu'},
                 {'source': 'two', 'reference': 'bi', 'hypothesis': 'bi', 'direction': 'ca2eu'}]
        validate_cached_predictions(saved, samples, 'ca2eu')
        for invalid in [saved[:1], list(reversed(saved)), [dict(saved[0], reference='changed'), saved[1]]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_cached_predictions(invalid, samples, 'ca2eu')

    def test_score_inputs_support_both_saved_result_formats(self):
        rows = [{'source': 'Español', 'hypothesis': 'Euskara', 'reference': 'Euskara'}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            for payload in [rows, {'per_sample': rows}]:
                source.write_text(json.dumps(payload))
                self.assertEqual(load_triples(source), [{'src': 'Español', 'mt': 'Euskara', 'ref': 'Euskara'}])
            source.write_text(json.dumps([{'source': 'missing fields'}]))
            with self.assertRaises(ValueError):
                load_triples(source)

    def test_comet_bridge_uses_worker_without_importing_comet_in_parent(self):
        # A deterministic scoring double exercises the real subprocess and JSON protocol.
        # It avoids downloading a checkpoint or pretending to validate model quality.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / 'model.ckpt'
            checkpoint.touch()
            (directory / 'comet.py').write_text('''from types import SimpleNamespace
def download_model(name):
    raise AssertionError("Local checkpoint should be used")
def load_from_checkpoint(path):
    class Model:
        def eval(self):
            pass
        def predict(self, data, batch_size, gpus):
            assert data == [{"src": "Español", "mt": "Euskara", "ref": "Euskara"}]
            assert batch_size == 8 and gpus == 0
            return SimpleNamespace(system_score=0.75, scores=[0.75])
    return Model()
''')
            with patch.dict(os.environ, {'PYTHONPATH': str(directory)}):
                evaluator = ExternalCometEvaluator(sys.executable, checkpoint=str(checkpoint))
                result = evaluator.predict([{'src': 'Español', 'mt': 'Euskara', 'ref': 'Euskara'}])
                self.assertEqual(result.system_score, 0.75)


if __name__ == '__main__':
    unittest.main()
