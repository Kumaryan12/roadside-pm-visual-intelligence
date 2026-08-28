# Environmental Modelling & Software submission checklist

## Completed in this package

- Research Article prepared in editable Elsevier `elsarticle` review format.
- Concise title, author name, corresponding-author marker and full postal affiliation.
- Self-contained 147-word abstract with no citations (journal maximum: 150 words).
- Five keywords (journal range: 1--7).
- Five mandatory highlights in a separate editable file; every bullet is at most 85 characters.
- Mandatory graphical abstract supplied as editable LaTeX, vector PDF and 1328 x 531 px PNG.
- Equations remain editable and numbered when referenced.
- Tables remain editable, use captions/notes, and contain no vertical rules or shaded cells.
- Figures 1--12 are collected under `figures_submission_final/`; calibration panels are named `Figure_10a` and `Figure_10b`.
- Vector charts are supplied as PDF; raster figures exceed normal print-size requirements.
- Figure captions are present in the manuscript and duplicated in `figure_captions.txt`.
- Author-year references use the Elsevier Harvard bibliography style.
- Data/software, ethics/data-governance, CRediT, competing-interest and funding statements included.
- Required generative-AI declaration included before the references.
- Acknowledgements included before the references.
- Grouped, assisted, diagnostic and exploratory results remain explicitly distinguished.
- Cover letter is tailored to *Environmental Modelling & Software*.
- Exact code and derived-result release archived under the concept DOI
  `10.5281/zenodo.22121571`; `v1.0.3-paper` is identified as the frozen analysis snapshot.
- Dataset-montage transformations, exact source filenames and image licences are documented.
- The graphical abstract is reproducible from editable TikZ/LaTeX source; no text-to-image or generative image model was used.
- EMS software metadata state the developer/contact, first release year, version, MIT licence, zero cost, Python/dependencies, archive size and supported systems.
- A separate editable Declaration of Competing Interest Word file is supplied.

## Author actions required before upload

- Confirm the no-specific-funding statement is accurate.
- Confirm the spelling and consent of acknowledged contributors: Sreejith Chakrapani and Navaneethakrishnan V.
- Add an ORCID in Editorial Manager if available.
- Confirm that the recorded MIT (TRAQID) and CC BY 4.0 (Taiwan) licences remain current at upload time; do not upload the raw source-image collections.
- Review, sign if required, and upload `declaration_of_competing_interest.docx`.
- Review and approve the generative-AI disclosure wording; disclose the same tools in the submission workflow when prompted.
- Enter author metadata, affiliation, funding and data-availability answers consistently in Editorial Manager.
- Select the appropriate data-linking option in the portal and attach the repository DOI.
- Visually inspect the final portal-generated proof, especially equations, tables, affiliations and figure order.

## Files to upload

1. `main.tex`, `references.bib`, `main.bbl`, and the complete `figures_submission_final/` folder (manuscript source package).
2. `main.pdf` or the final compiled candidate PDF for reviewer convenience.
3. `highlights.txt` as **Highlights**.
4. `graphical_abstract.pdf` as **Graphical abstract**; retain the exact-size PNG as backup.
5. `figure_captions.txt` only if the portal asks for a separate legend file.
6. `declaration_of_competing_interest.docx` as the separate Declaration of Competing Interest.
7. Optional supplementary/provenance files if the editor requests the extensive appendices separately.

## Scope and reporting checks

- Sex/gender reporting is not applicable: no participant-level biological or social variables are analysed.
- No jurisdictional map is included in the current manuscript.
- No chemical source-apportionment claim is made.
- The source datasets are public, but public availability does not automatically permit redistribution of their imagery.
- The GitHub repository and immutable Zenodo DOI are both stated in the manuscript.

## Length note

The review PDF is intentionally double-spaced and includes extensive appendices. No formal page limit requires separation, so the appendices remain in the main submission for auditability. If the editor requests a shorter article, move the provenance, per-date and sensitivity appendices into a supplementary PDF without changing reported values.
