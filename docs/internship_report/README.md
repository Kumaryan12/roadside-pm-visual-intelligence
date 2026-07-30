# Internship report

This folder contains the complete LaTeX internship report and reproducible
figures for the Roadside PM Visual Intelligence project.

## Files

- `internship_report.tex` - complete editable report.
- `generate_figures.py` - regenerates all charts and architecture diagrams from
  project artifacts and audited result values.
- `figures/` - report figures, including a real three-lens MUMMA sample.
- `Makefile` - convenience build commands.

## Before submission

Edit the identity macros near the top of `internship_report.tex`:

- `\studentid`
- `\degreeprogramme`
- `\institution`
- `\hostorganisation`
- `\academicsupervisor`
- `\industrysupervisor`
- `\internshipperiod`

The student name is prefilled as `Aryan Satyendra Kumar` and can also be edited.

## Build

```bash
cd docs/internship_report
python3 generate_figures.py
make
```

The built PDF is copied to `../../output/pdf/roadside_pm_internship_report.pdf`.

## Scientific reporting note

The report deliberately separates overlap-contaminated random diagnostics from
whole-date holdout results. The primary latest MUMMA result is the
validation-selected MERRA-2 nested ensemble (pooled R2 0.538). The fixed 50:50
ensemble result (pooled R2 0.566) is retained as a diagnostic, not presented as
the primary selected model.
