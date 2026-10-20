# multimodal-informal-settlement-extraction
Code accompanying the study on Sentinel-2 and Google Street View multimodal fusion for informal settlement extraction in Nairobi.

# Multimodal Informal Settlement Extraction

Code accompanying the manuscript:

"Reducing Uncertainty: The Role of Street-View and Satellite Imagery in Multimodal Informal Settlement Extraction"

## Overview

This repository contains the code used for data preprocessing, multimodal model
training, evaluation, prediction-level interpretation, leave-one-out analysis,
Grad-CAM analysis, semantic analysis, and full-area prediction.

## Repository structure

- preprocessing/: spatial alignment and reference preparation
- model_pipeline/: feature extraction, model training, and final evaluation
- interpretation/: ΔGSV, LOO, Grad-CAM, and semantic analyses
- full_area_analysis/: full-area prediction and GSV influence analysis
- figures_tables/: publication figures and tables
- audits/: data-availability and spatial-reference checks

## Data availability

Sentinel-2 imagery is publicly available from the European Space Agency.

Reference labels used in the study were provided by the IDEAtlas group and are
not redistributed in this repository.

Google Street View imagery was accessed through the Google Maps Platform API
and is not redistributed because of licensing restrictions.

## Reproducibility notes

The code reflects the computational workflow used in the study. Some scripts
retain the file paths and computing-environment settings used during the
original analysis on local and HPC systems. Users may need to adapt these paths
to their own environment.

Because some input datasets cannot be redistributed, the repository is intended
to document and reproduce the analytical workflow when the required input data
are available.

## Citation

Citation information will be added after publication.

## License

MIT License.
