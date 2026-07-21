# facedyn

A Python toolkit for interpretable analysis of facial Action Unit (AU) time
series: temporal smoothing, dimensionality reduction (NMF), interpretable
time-series feature extraction (CMFTS), and classification — usable for any
condition comparison (real/fake, emotion, patient/control, ...), not just
the deepfake-detection use case it was built to reproduce.

Originally a port of the R analysis featured in Murphy, Cook & Cuve,
"Interpretable facial dynamics as behavioral and perceptual traces of
deepfakes" (in prep).

Project is under active development.

## Development install

```bash
pip install -e ".[dev]"
pytest
```
