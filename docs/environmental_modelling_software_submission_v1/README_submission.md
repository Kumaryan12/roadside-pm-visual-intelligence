# Submission package

Target journal: **Environmental Modelling & Software**
Article type: **Research Article**

## Compile locally

From the repository root:

```bash
mkdir -p build/ems_submission
tectonic --outdir build/ems_submission \
  docs/environmental_modelling_software_submission_v1/main.tex
```

The compiled manuscript is written to `build/ems_submission/main.pdf`.
Tectonic resolves the BibTeX bibliography automatically.

## Principal files

- `main.tex`: editable Elsevier manuscript source.
- `references.bib`: editable bibliography database.
- `main.pdf`: locally compiled review manuscript.
- `figures_submission_final/`: final numbered Figures 1--12 used by `main.tex`.
- `figures_submission/`: preserved original colour figures for future revision.
- `figure_captions.txt`: figure legends for portals that request a legend file.
- `highlights.txt`: mandatory 3--5 article highlights.
- `graphical_abstract.tex`: editable graphical-abstract source.
- `graphical_abstract.pdf`: mandatory vector graphical abstract.
- `graphical_abstract.png`: exact-size 1328 x 531 px backup.
- `declaration_of_competing_interest.docx`: separate editable declaration.
- `cover_letter.md`: editable cover letter.
- `submission_checklist.md`: completed requirements and remaining author actions.

## Before submission

The exact code and derived-result snapshot is archived at
<https://doi.org/10.5281/zenodo.22121571>. Complete the remaining author actions
in `submission_checklist.md`.
