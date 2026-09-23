# Local review guide

1. Read `REVIEW_STATUS.md`. Studio and neural factory are separate execution paths.
2. Run `python run.py --no-browser` on loopback. Visit `/` and the API documentation
   at `/docs`. Build from specimen data and inspect the complete scorecard.
3. Read `evidence/benchmark.json`. `python -m modelrig.review prepare-sms` recreates
   the public corpus and provenance; `benchmark` repeats the experiments.
4. Inspect a neural build's `training_record.json`, `metadata.json`,
   `eval_report.json`, `hf_model.json`, adapter and deploy directories. Run
   `python -m modelrig.review predict --path ARTIFACT --text TEXT` locally.
   In the extracted review ZIP, ARTIFACT is `sms_specialist`. Install `.[api,ml]`
   first; add it to the local model list with
   `python -m modelrig.review import-artifact --path sms_specialist`.
5. Run the tests. `tests/test_review_readiness.py` checks evidence integrity,
   dataset identity, recall and tensor tampering.
6. Compile `paper/majestic_research.tex` with `pdflatex` twice or Tectonic.
   The bibliography is embedded; no external image or `.bib` file is needed.

Test counts are software evidence. An interval for SMS accuracy is not a safety,
privacy, business or mobile-device certificate. The strongest honest review is a
working specialist, its evidence, and a concrete account of the remaining work.

## Confidentiality

The patent PDF stays at its original local path. Its local extraction and source
digest are in ignored `private/`; they are not training data, paper sources or
package contents. No repository push, upload, email, publication or paper
submission was performed. The paper is a private review manuscript until the
owner chooses to circulate it. Ignore rules protect ordinary staging, not an
explicit forced add, and are not encryption.
