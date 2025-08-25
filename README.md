# Semiconductor Wafer Test Data Anomaly Detection
This project develops an anomaly detection system using Python to identify drifting behavior in post-silicon test data. The system helps engineers flag problematic test parameters and pinpoint the test sites responsible for anomalies, improving the accuracy and efficiency of post-silicon validation.

## 01_initial_model
Initial development of machine learning models using Niki's summary spreadsheet, which includes labels for whether each test parameter should be investigated/released.
- Input: Median and standard deviation of each test parameter across 192 test sites
- Output: Predicts whether a parameter should be labeled as investigate or release

## 02_raw_data_processing
We discovered that Niki's summary spreadsheet **did not distinguish between NaN and 0 values**. Both were treated as 0. This misrepresentation was corrected by referencing the raw data and **automating the preprocessing pipeline** to preserve the distinction.

## 03_raw_binary_model
Retraining of the binary classification model using the corrected raw data and updated labels. This version reflects improved data integrity and model performance. Trained models were exported as joblib files to be integrated into Exensio-MA.

## 04_site_model
Once problematic test parameters are flagged, we identify the specific test sites responsible using an unsupervised Isolation Forest model.
Performance Highlights:
- Successfully identified all problematic sites for **97% of flagged test parameters**.
- For the remaining 3%, the model uncovered labeling inconsistencies and pointed out several human judgement errors.

## 04_site_model
Combined all data preprocessing, feature augmentation, and model prediction processes into one single script that could be uploaded into Exensio-MA for quick deployment.

## joblib files
All .joblib files in this repository are pre-trained models and are ready for download and integration into other projects. These files allow for quick deployment without retraining.
