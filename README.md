# Roadside PM Visual Intelligence

This repository develops a physically interpretable visual-feature pipeline for roadside particulate density estimation using traffic-camera imagery, vehicle detection, road-dust indicators, temporal features, and emission-aware traffic proxies.

## Research Direction

The project does not treat raw images as direct PM sensors. Instead, it extracts physically meaningful visual proxies from roadside camera imagery:

- vehicle counts and vehicle composition
- heavy vehicle contribution
- auto-rickshaw-aware traffic features
- road dust indicators
- haze and visibility indicators
- dust-traffic interaction features
- temporal persistence features
- optional environmental and OSM context

## Main Research Question

Can roadside traffic-camera imagery be converted into physically meaningful visual indicators that improve estimation of PM2.5 or effective particulate density?

## Repository Structure

```text
configs/        Configuration files
data/           Raw/intermediate/processed data, not committed
src/            Source modules
scripts/        Reproducible pipeline scripts
notebooks/      Analysis notebooks
outputs/        Generated outputs, not committed
audits/         Manual audit templates
docs/           Methodology and research notes
