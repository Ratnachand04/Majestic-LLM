# Majestic — technical review brief

**Current deliverable:** a local specialist-building research system, a trained
transformer classifier, reproducible measurements, and a private research paper.
The broader model compiler described in the design documents remains a research
and integration program. This is a technical diligence package, not a claim of
commercial traction or complete product readiness.

## Demonstration

Start the local studio with `python run.py`. The model list includes the trained
neural SMS specialists; click **Use** and submit a message in the inference panel.
The specimen-data build demonstrates the compiler with a small TF-IDF model.
The analytical Qwen plan is labeled separately from the model that executes.

The public-data experiment covers 5,159 deduplicated SMS inputs. The initial
DistilBERT LoRA build uses 4,128 training and 1,031 held-out examples. It reaches
99.13% accuracy and 97.99% macro-F1, with a 95% Wilson accuracy interval of
98.35–99.54%. These are historical-dataset results, not field-performance claims.
Replicates and quantized-backend observations are in `evidence/neural.json`.
Across three fixed seeds, LoRA averages 98.87% accuracy and 97.36% macro-F1.
One QLoRA/NF4 deployment scores 99.13% accuracy and occupies 73.84 MB versus
134.87 MB for the unquantized deployment. It changes two of 1,031 predictions
relative to its own merged checkpoint. This is a compression result, not a
measured inference-speed or mobile-memory claim.

## What a reviewer can inspect

- An actual pretrained transformer, learned adapter, merged model and local inference.
- Hash-bound artifacts, explicit execution provenance, recall checks and held-out reports.
- Reproducible TF-IDF baselines across three fixed splits, including negative
  quality and runtime results under quantization.
- Typed specifications, auditable feasibility decisions, and graph analysis tests.
- A research manuscript with methods, results, limitations and primary references.

## Product thesis to test

Organizations may value a reproducible path from a narrow task and available data
to a local specialist with visible limitations. The proposed differentiation is
the integration of specifications, resource-aware refusal and deployment evidence.
Training algorithms, adapter serving and compiler representations have substantial
prior art. Market demand, customer retention and pricing have not been measured.

## Milestones before broader diligence claims

| Milestone | Acceptance evidence |
|---|---|
| First customer domain | Rights-cleared corpus, independent final labels, agreed error costs and acceptance bar |
| Common build path | Planner choices execute exactly in the neural backend; no unsupported recipe masquerades as a success |
| Device deployment | Actual artifact runs on the named target, with sustained latency, memory, power and thermal logs |
| Predictive planning | Boundary experiments test refused as well as accepted builds; stratified refusal precision/recall |
| Operational deployment | Authentication, tenant isolation, transactional registry/jobs, backups and incident/recall exercises |
| Research validation | Independent taxonomy coding, data ablations, field-like flip study and build-history evaluation |

Read `REVIEW_STATUS.md` for the subsystem-by-subsystem gaps. No fundraising amount,
revenue, valuation, patent grant, third-party approval or customer deployment is asserted.

## Private material

The patent source is kept locally and is excluded from this review archive and
paper attachments. This package was created on disk; it has not been published
or sent to any investor or external service.
