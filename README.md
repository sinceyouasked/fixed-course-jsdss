# Measure your course before you change it

Data and code for **"Measure your course before you change it: what three championship venues show about scoring, weather, setup and architecture"** (J. Mohiuddin, 2026; under review at the *Journal of Sports Economics*, submitted 18 November 2026).

This repository began as the reproducibility package for an earlier, single-venue version of the paper (*A fixed course under a moving game*, Augusta National only; `index.qmd`, `data/hole_year_panel_1983_2026.csv`, `scripts/analysis.py`). The current version adds Pebble Beach, Riviera and the green surveys. Everything the manuscript's tables and figures rest on is here or at the OSF deposit (https://doi.org/10.17605/OSF.IO/3HGJ2), which mirrors this repository and also holds the preprint PDFs.

## Layout

```
data/
  hole_year_panel_1983_2026.csv        hole-year panel, Masters 1983-2026 (main text, Sections 5-6)
  masters_fsa_first36_1934_2026.csv    first-36-hole field scoring average, 1934-2026 (Appendix E)
  fig_aug_series.csv, fig_aug_weather.csv   series behind Figures 1-3
  augusta/
    masters_hole_rounds_1983_2026.csv  hole-by-hole scoring by round, 1983-2026 (Figure 2 heat map)
    masters_players_1983_2026.csv      per-player rounds with world-ranking at the time
    masters_hole_locations_2016_2026.csv  hole locations transcribed from the published daily sheets (Section 6.6 input)
    bootstrap_cis.csv                  wild-cluster bootstrap intervals, Table 3
  pebble/
    pebble_players_2004_2026.csv       Data Golf per-player rounds by course, AT&T and U.S. Opens (Appendix B)
    pgatour_pebble_rounds_1947_2026.csv  PGA Tour past-results archive, final round at Pebble Beach 1947-2026
    pebble_sunday_series_1958_2026.csv Sunday field average, 61 usable years
  riviera/
    riviera_laopen_rounds_1926_2026.csv  per-player rounds, LA Open / Genesis, 1926-2026 (Appendix C)
    riviera_fsa_series_1926_2026.csv   field scoring average series
    riviera_series_weather_1973_2026.csv, riviera_weather_windows.csv,
    riviera_lcd_obs_playinghours.csv   NOAA LCD hourly observations, Santa Monica / LAX, playing hours only
    riviera_year_effects_playerFE_0.csv  year effects with player fixed effects
    riviera_championships_raw.txt      source notes for championship years
  greens/
    pebble_greens_18.csv               per-green area and slope shares, Pebble Beach (Appendix D, Table D1)
    riviera_greens_18.csv              per-green area and slope shares, Riviera
    augusta_green_areas_estimated.csv  Augusta per-green estimates from published StrackaLine shares
    blacksheep_golflogix_vs_ezlocator.csv  vendor calibration: GHIN/GolfLogix heat map against ezLocator laser survey, 7 greens
    *_segmentation_check.png           green-outline masks used for the area calibration, for inspection
scripts/
  analysis.py, stacked.py, checks.py, fig_lib.py   Augusta hole-year panel (unchanged from the JSDSS version)
  pebble_riviera/                      Pebble Beach and Riviera series, weather model and figures
  appendix/                            aging / prime-age robustness, tour panel, signature stability
results/
  *_results.txt                        script output as quoted in the paper
  fig2_hole_heatmap.png                Figure 2 as submitted
  green_size_comparison_with_augusta.png  Appendix D figure
```

## Green surveys (Appendix D)

Pebble Beach and Riviera slope shares were read from GHIN/GolfLogix green heat maps as follows: the legend bar (0-7% ramp) is sampled to a colour look-up table; the green surface is masked; arrows and labels are filled by nearest surrounding value; the share of surface at each threshold is the share of masked pixels whose nearest legend colour falls at or below it. Areas were calibrated from the GPS back-to-front yardage, with published figures substituted for five Pebble Beach greens where the diagonal orientation made the yardage calibration unreliable (flagged in the `source` column). The same procedure was run on six Black Sheep greens and compared with the club's ezLocator laser survey to estimate the heat-map bias at each threshold (`blacksheep_golflogix_vs_ezlocator.csv`; the 27th is included but excluded from the calibration because the GHIN polygon merges the practice green). The paper reports the 4% threshold, where the bias is smallest (+3.5 points, SE 2), and uses the 3% threshold only laser-against-laser. The Black Sheep survey summary is reproduced with the club's permission; the full ezLocator report is the club's property and is not deposited.

## Reproducing

Python 3.11; `pip install -r requirements.txt` (pandas, numpy, statsmodels, scipy, matplotlib). Each script reads from `data/` by relative path and writes to `results/`. The Augusta scripts are run from the repository root as before; the Pebble/Riviera scripts from `scripts/pebble_riviera/`.

## Sources

Augusta National Golf Club, *2026 Masters Media Guide*; Masters.com hole-by-hole archive; Data Golf; PGA Tour past results; NOAA GHCN-Daily (Augusta Bush Field, Monterey) and Local Climatological Data (Santa Monica, LAX); GHIN/GolfLogix green maps; StrackaLine (Augusta green shares as published); ezLocator survey courtesy of Black Sheep Golf Club. Each is cited in full in the manuscript's reference list.

## Licence

Code: MIT. Data compiled by the author: CC BY 4.0. Underlying scores and weather are the property of their publishers and are reproduced here for research use.

Junaid Mohiuddin · ORCID 0009-0009-8806-4853 · jgmohiuddin@gmail.com
