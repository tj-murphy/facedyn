# Input Data Format

This describes the minimal shape your data needs to be in to use `facedyn`.
The core pipeline (smoothing, normalisation, NMF, representative-AU
selection, CMFTS feature extraction) makes **no assumption about what your
comparison is** — real-vs-fake, emotion categories, patient-vs-control,
anything. It only assumes a specific *shape*: long-format, one row per
video-frame, with AU intensity columns.

## Minimal required shape

One row per (video, frame). At minimum:

| Column | Purpose | Default name | Configurable? |
|---|---|---|---|
| Video identifier | groups frames belonging to the same video | `video_filename` | Yes — every function that needs it takes a `video_col` parameter |
| AU intensity columns | one or more numeric columns to analyze | any column matching `_r$` (e.g. `AU01_r`) | Yes — `RollingSmoother(column_pattern=...)` or pass `columns=[...]` explicitly |

Example, minimum viable input:

| video_filename | frame | AU01_r | AU06_r | AU12_r |
|---|---|---|---|---|
| vid_001 | 1 | 0.4 | 0.1 | 0.0 |
| vid_001 | 2 | 0.5 | 0.2 | 0.1 |
| vid_002 | 1 | 0.0 | 0.3 | 0.6 |

That's sufficient to run `RollingSmoother` and (once implemented)
normalisation, NMF, and CMFTS feature extraction — none of these steps need
to know what your videos represent or what you're comparing.

Column-naming conventions in this package (like the `_r` suffix) come from
OpenFace's AU intensity output format, since that's what the original study
used — but every function accepts overrides, so you are not required to use
OpenFace or match its naming.

## Splitting into train/test

Once you're ready to split data for model training, pick the splitter that
matches your study design (see `src/facedyn/splitting.py`):

- **`group_train_test_split`** (recommended default) — needs only the video
  identifier column above. Keeps each video's frames together on one side
  of the split; makes no assumption about pairing or class structure. Use
  this unless you specifically need matched pairs.
- **`paired_train_test_split`** — for designs with explicit 1:1 matched
  pairs that must always land on the same side of the split (e.g. this
  package's original real/fake deepfake pairing). Needs two additional
  columns:

  | Column | Purpose | Default name |
  |---|---|---|
  | Label | distinguishes the two roles in a pair (e.g. real/fake) | `isfakeorreal` |
  | Pair pointer | for a row with the "primary" label (e.g. real), gives the `video_filename` of its paired counterpart | `corresponding_video` |

  If your data doesn't have an explicit matched-pairs structure, use
  `group_train_test_split` instead — don't fabricate a pairing column just
  to satisfy this function.

## Downstream steps (classification, feature interpretation)

Whatever "condition" you want to detect differences in — real/fake,
emotion, or otherwise — becomes the label column you pass to the eventual
classifiers (not yet implemented; see `PIPELINE.md`). The interpretable
features (NMF components, CMFTS metrics) are computed independently of that
label, so the same feature-engineering pipeline supports any downstream
comparison you define.
