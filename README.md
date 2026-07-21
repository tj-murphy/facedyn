# facedyn

Python implementation of the interpretable facial-dynamics deepfake
detection pipeline from Murphy, Cook & Cuve, "Interpretable facial dynamics
as behavioral and perceptual traces of deepfakes." Ports the original R
analysis into a reusable, tested package built on scikit-learn-style
transformers.

Project is under active development. See:

- [`PIPELINE.md`](PIPELINE.md) — the pipeline steps being implemented and
  their porting status
- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) — package layout and
  design rationale

## Development install

```bash
pip install -e ".[dev]"
pytest
```
