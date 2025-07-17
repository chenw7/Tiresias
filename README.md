# Project Tiresias
This repository contains the work for Project Tiresias, an initiative to automate the analysis of manufacturing test cases using machine learning.

## Current Status: Parameter-Level Classification
The project's first phase involved creating a binary classification model to predict whether a given test parameter contains anomalous site data. This phase has been successfully completed.

The best-performing model, LightGBM, achieved the following results:
 Accuracy: 97.3%
 Precision: 91.0%
 Recall: 91.0%

## Next Steps: Site-Level Anomaly Detection
With a reliable parameter-level classifier, the project is now moving to its second phase: pinpointing the specific problematic sites within the parameters that were flagged for investigation.

To accomplish this, we are implementing an Isolation Forest model to perform anomaly detection on the individual sites for each flagged parameter.
