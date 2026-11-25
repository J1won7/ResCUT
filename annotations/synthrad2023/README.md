# SynthRAD2023 Evaluation Annotations

Thirty pelvic CBCT cases from the official Task 2 validation release:

- `artifact/`: 30 full binary artifact masks, including 11 empty masks.
- `gluteal/`: 30 merged gluteal-muscle reference masks used in evaluation.
- `body/`: 30 fixed source-derived body masks used for preservation metrics.
- `rois/`: frozen 16x16 square coordinates for both regions on eligible slices.
- `metadata.json`: grid geometry, input-preparation parameters, mask hashes and
  annotation status.
- `manifest.json`: editable evaluation manifest; add predictions under `images`.

These masks use the evaluation grid, not necessarily the original downloaded
grid. Run `python -m evaluation.prepare_synthrad` before inference. No CBCT or CT
images, restoration outputs, or checkpoints are distributed here. Empty masks
mean no annotated artifact, not a missing download. Do not remove them from
preservation evaluation or assign zero artifact scores to them.

The original data are distributed by the SynthRAD2023 organizers:
https://doi.org/10.5281/zenodo.7868168 . Use is subject to the official dataset
terms; the code's license does not relicense the underlying data. This package
does not make a new license grant for the annotations. Retain the dataset and
study attributions when redistributing derived annotations.

Dataset reference: Thummerer et al., Medical Physics 50 (2023), 4664-4674,
doi:10.1002/mp.16529.
