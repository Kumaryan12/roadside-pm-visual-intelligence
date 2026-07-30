# Particle density and source-proxy pipeline

Canonical objective: model OPC size distributions and effective particle
density, then estimate non-negative source-related PM2.5 components with mass
closure and an explicit unresolved component.

Until chemical source labels exist, outputs must be described as source-proxy
estimates rather than chemically validated source attribution.

The current-data exploratory entry point is:

```bash
python -m pipelines.particle_source_attribution.run_source_proxy --help
```

It fits NMF only on the training PM/OPC rows, then predicts the inferred
component percentages from non-PM contextual features. Random-row results are
diagnostic; `--protocol date --test-date YYYY-MM-DD` is the fair mode.

For explicitly assumption-conditioned scenario analysis with Monte Carlo
uncertainty, use:

```bash
python -m pipelines.particle_source_attribution.run_assumption_scenarios --help
```

The versioned illustrative priors are stored under
`configs/source_attribution/`. They must be replaced rather than silently
retuned when measured source profiles become available.

The scientifically preferred current-data analysis models directly observed
particle-size regimes rather than named sources:

```bash
python -m pipelines.particle_source_attribution.run_particle_regimes --help
```
