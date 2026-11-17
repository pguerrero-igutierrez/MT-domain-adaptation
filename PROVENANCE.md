# Clinical corpus and evaluation versions

The release contains two distinct clinical evaluation sets. Historical predictions and numerical results have been preserved.

| Artifact | Provenance | Cases |
| --- | --- | --- |
| `outputs/test_set_clinical_reported.json` | Ordered sources and references recovered from all four agreeing clinical prediction caches | 66 |
| `outputs/test_set_clinical.json` | Alternative split corresponding to the currently released corpus | 79 |

Regenerate the first file with `python scripts/prepare_reported_clinical_test.py`. It contains only inputs and references already present in the public prediction caches, without model hypotheses.

## Corpus checksums

A local archived file named `eu-clinical_backtranslated.json` has SHA-256:

```text
2af0fe8ddc11d664e5d1e15d30dea711049b938af40f211c5ccde9ef083ee6a3
```

It contains 1,729 raw records. Keeping source/target strings of at least 20 characters and a maximum bidirectional character-length ratio of 3 leaves 1,305 records. Shuffling with Python `random.seed(42)`, then taking 90% train and 5% validation, produces **1,174 training, 65 validation and 66 test cases**. The complete ordered test sources and references match all four published clinical prediction caches exactly.

The current public `backtranslated-corpus/eu-clinical_eu2ca.json` has SHA-256:

```text
1819d0e179f979f37279fd71c8dc2d166d664f04e2008ceb08ec806fa50e9b15
```

It also contains 1,729 raw records, but the same filtering leaves 1,569 records and produces a **1,412/78/79 split**. It is a different version and is retained without modification.

The archived version has not been added to the public repository in this cleanup. Its distribution needs to be resolved with the authors' pending licensing/data-release decision. The matching split is evidence identifying a relevant archive, not independent proof of which bytes every historical training job read or of train/test independence. The archived code/model revisions and token-matching details remain subjects for the manuscript review.

## Choosing the corpus

Steps 09b, 10, 12 and clinical split reconstruction in step 13 accept `--clinical-data /path/to/corpus.json`. Use that option when an authorized copy of the archive is available. Default inputs use the currently public corpus and should be described as new runs on that version.

For reuse of the reported evaluation predictions, pass `--test-file outputs/test_set_clinical_reported.json`. This does not require redistributing the archived training corpus. Existing caches are accepted only for the same ordered sources and references.
