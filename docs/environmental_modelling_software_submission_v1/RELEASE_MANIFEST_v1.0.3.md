# v1.0.3-paper release manifest

This document defines the frozen analysis snapshot associated with the submitted
manuscript. The release retains the Zenodo concept DOI
`10.5281/zenodo.22121571`; `v1.0.3-paper` is the versioned GitHub analysis
snapshot.

## Submission source

- `main.tex`, `references.bib`, and the generated `main.bbl`
- compiled 56-page `main.pdf`
- Figures 1--12 in `figures_submission_final/`, including separate calibration
  panels `Figure_10a` and `Figure_10b`
- editable and rendered graphical abstract
- highlights, cover letter, figure captions, competing-interest declaration,
  dataset-image provenance, and submission checklist

## Frozen analytical outputs

- Taiwan chronological predictions and split audit
- Taiwan network-reference ablation predictions
- Taiwan reviewer-requested datewise, outage, and 5,000-resample paired
  date-bootstrap outputs
- TRAQID chronological calibration predictions and per-date metrics
- TRAQID reviewer-requested grouped-sensitivity and calibration audit tables

The release archive contains a `SHA256SUMS` file covering every packaged file.
Raw public-dataset images and trained model checkpoints are excluded because
they are distributed separately by their original providers or are readily
regenerated. No `.DS_Store`, `.aux`, `.log`, `.blg`, or `.out` files are part of
the archive.

## Figure provenance

Figures 1 and 3--12 and the graphical abstract were produced through
deterministic author-verified code/LaTeX workflows. Figure 2 contains attributed
public-dataset frames. Those frames were uniformly converted to grayscale and
cropped for layout; a uniform contrast adjustment was applied, and no objects
or semantic content were added or removed. See
`dataset_scene_figure_sources.md` for the exact source files and licences.
