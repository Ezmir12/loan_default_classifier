# Loan Default Prediction

Predicts whether a loan will be paid off or sent to collections, comparing
three classifiers on IBM's loan repayment dataset:

- K-Nearest Neighbors (k tuned via 5-fold cross-validation on the training
  set, so the test set stays untouched until final evaluation)
- Logistic Regression
- Decision Tree

The training set is imbalanced toward paid-off loans, so SMOTE is used to
rebalance it before fitting. Each model is evaluated on accuracy, precision,
recall, F1, and ROC-AUC, with confusion matrices, a decision tree diagram,
and feature-importance/coefficient plots generated for inspection.

## Simulated retraining

`ModelRetrainer` simulates what a periodic retraining job would do: draw a
fresh batch of augmented samples (existing rows resampled with noise),
merge them into the training set, rebalance with SMOTE, and refit every
model. It runs a few cycles inline in the notebook to show how accuracy
shifts as the training data grows — it does not run on an actual schedule
(no cron job, no background process). Wiring it to run periodically in
production is future work, not something this notebook does.

## Files

- `loan_default_classifier_and_retraining.ipynb` — full pipeline: data loading,
  preprocessing, training, evaluation, and the simulated retraining cycles
- `Report.pdf` — write-up of the approach and results

## Setup

```bash
pip install -r requirements.txt
```

The notebook pulls its dataset from a public IBM Cognitive Class CDN link —
if that link is ever taken down, point `url` in the first data-loading cell
at a local copy of `loan_train.csv` instead.
