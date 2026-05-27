# Methodology

## Core Principle

This project does not directly treat RGB images as PM sensors. Instead, it extracts physically interpretable visual proxies from traffic-camera imagery and uses them for PM2.5 or effective particulate density estimation.

## Main Feature Families

1. Traffic features
2. Vehicle composition features
3. Auto-rickshaw-aware traffic features
4. Road dust and haze features
5. Dust-traffic interaction features
6. OSM/geospatial context
7. Temporal persistence features
8. Optional environmental features

## Validation

All model evaluations should use timestamp-grouped validation to avoid leakage between frames from the same timestamp.
