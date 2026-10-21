# A fixed course under a moving game

Junaid Mohiuddin · ORCID 0009-0009-8806-4853 · jgmohiuddin@gmail.com

Manuscript submitted to the *Journal of Statistics and Data Science in Sports* (JSDSS), built with the journal's Quarto manuscript template (`quarto-journals/jasa` extension).

## Reproduce

```
pip install -r requirements.txt
quarto render            # HTML + PDF into _manuscript/
```

The six figures are drawn at render time by `scripts/fig_lib.py` from the four files in `data/`. Every number in the text comes from:

```
cd scripts
python analysis.py       # year-level inference, panel, event study, bootstraps (~20 min)
python stacked.py        # stacked clean-control estimator
python checks.py         # reconciliation and sensitivity checks
```

Output of the last run is in `scripts/analysis_results.txt`.

## Data

| File | Content |
|---|---|
| `data/masters_fsa_first36_1934_2026.csv` | Field scoring average relative to par, all 90 Masters, headline and cut-free first-36 series |
| `data/fig_aug_weather.csv` | Playing-hours temperature, wind and related variables from NCEI LCD station USW00003820, 1948–2026, with FSA |
| `data/fig_aug_series.csv` | Published tournament yardage by year |
| `data/hole_year_panel_1983_2026.csv` | 756 hole-years: hole score relative to par, yardage, and dated architectural changes |

The same files, the course-change chronology and the preprint of the companion study are deposited at the Open Science Framework, https://doi.org/10.17605/OSF.IO/3HGJ2.

## Licence

Text and figures CC BY 4.0; code MIT; data CC0.
