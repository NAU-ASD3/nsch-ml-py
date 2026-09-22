# prevalence

Autism prevalence computed straight from the Census Bureau's NSCH public-use
files, with design-based standard errors. This module reads the raw Stata
files and does not go through the harmonization pipeline, so it doubles as an
independent check on that pipeline. `notebooks/autism_prevalence_2016_2024.py`
is the worked example, and `analyses/autism_prevalence.py` writes the tracked
results table under `analyses/results/`.

::: nsch_ml.prevalence
    options:
      members:
        - Estimate
        - find_topical_file
        - read_topical_year
        - classify_autism
        - add_indicators
        - load_children
        - weighted_proportion
        - weighted_total
        - prevalence_row
        - prevalence_table
        - read_do_labels
        - sha256_of
