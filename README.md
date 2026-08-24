# House Price Prediction

A Streamlit application that predicts house prices from square footage,
bedrooms, bathrooms, year built, and a neighborhood rating.

## What's new in this version

- **800-row dataset** instead of 5 rows (see `Dataset/generate_dataset.py`) —
  the original dataset was too small for any model to learn a real pattern.
  Swap this for a real housing dataset when you have one.
- **Fixed a path bug**: the training script previously looked for the CSV in
  a `Dataset/` folder that didn't exist relative to where it searched.
- **Model comparison + evaluation**: trains Linear Regression, Random
  Forest, and Gradient Boosting, splits into train/test, and automatically
  keeps whichever scores best on held-out data (reports R², MAE, RMSE).
- **Feature engineering**: derives `HouseAge` from `YearBuilt`, which is
  usually more predictive than the raw year.
- **The app now shows**: an error margin alongside the prediction, price
  per square foot, a feature-importance chart, model accuracy metrics, and
  a batch-prediction tab (upload a CSV, get predictions back as CSV).
- **Faster reloads**: the model is cached with `st.cache_resource` instead
  of being reloaded from disk on every click.
- Saved with `joblib` instead of raw `pickle` (sklearn's recommended format).

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the dataset (or replace Dataset/house_data.csv with real data)
python Dataset/generate_dataset.py

# 2. Train the model
python Model/train_model.py

# 3. Run the app
streamlit run app.py
```

## Project structure

```
├── Dataset/
│   ├── generate_dataset.py   # synthetic data generator
│   └── house_data.csv        # training data (generated)
├── Model/
│   ├── train_model.py        # trains, evaluates, saves best model
│   ├── house_model.joblib    # trained model (generated)
│   ├── metrics.json          # accuracy metrics (generated)
│   └── feature_importance.json
├── app.py                    # Streamlit UI
└── requirements.txt
```

## Next steps for even better results

- Swap the synthetic dataset for real housing data (e.g. county assessor
  records, Zillow/Redfin exports, or the Ames Housing dataset on Kaggle).
- Add categorical features like ZIP code or school district, one-hot or
  target-encoded.
- Try `GridSearchCV`/`RandomizedSearchCV` for hyperparameter tuning once
  you have enough real data to justify it (with only a few hundred rows,
  tuning tends to overfit to the validation split).
- Log-transform `Price` before training if your real data is skewed
  (housing prices usually are).
