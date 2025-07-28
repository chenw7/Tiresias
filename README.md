# Post-Silicon Test Data Anomaly Detection
This project develops an anomaly detection system using Python to identify drifting behavior in post-silicon test data. The system helps engineers flag problematic test parameters and pinpoint the test sites responsible for anomalies, improving the accuracy and efficiency of post-silicon validation.

## 01_initial_model
Initial development of machine learning models using Niki's summary spreadsheet, which includes labels for whether each test parameter should be investigated/released.
Input: Median and standard deviation of each test parameter across 192 test sites
Output: Predicts whether a parameter should be labeled as investigate or release
Best Model: **XGBoost with 96.9% accuracy**, outperforming Random Forest, LightGBM, and Decision Tree

## 02_ground_truth_fix
After analyzing discrepancies between model predictions and Niki's judgments, we revised some labels to improve consistency. This step acknowledges the subjectivity in the original labeling process.
We also identified a heavy class imbalance, prompting a **switch from accuracy to F1 score** as the primary evaluation metric to better reflect model performance in imbalanced scenarios.

## 02.5_debugging
We discovered that Niki's summary spreadsheet **did not distinguish between NaN and 0 values**. Both were treated as 0. This misrepresentation was corrected by referencing the raw data and **automating the preprocessing pipeline** to preserve the distinction.

## 03_raw_binary_model
Retraining of the binary classification model using the corrected raw data and updated labels. This version reflects improved data integrity and model performance.
**Best Model: XGBoost**
**- Accuracy: 97.6%**
**- Precision: 91.9%**
**- Recall: 92.6%**
**- F1 Score: 92.2%**

## 04_site_model
Once problematic test parameters are flagged, we identify the specific test sites responsible using an unsupervised Isolation Forest model.
Performance Highlights:
- Successfully identified all problematic sites for **97% of flagged test parameters**.
- For the remaining 3%, the model uncovered labeling inconsistencies, suggesting potential human judgment errors.

## pkl files
All .pkl files in this repository are pre-trained models and are ready for download and integration into other projects. These files allow for quick deployment without retraining.
