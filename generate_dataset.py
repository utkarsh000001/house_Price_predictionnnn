"""
Generates a larger, more realistic synthetic housing dataset.

The original dataset had only 5 rows, which is not enough to train or
evaluate any model meaningfully. This script produces 800 rows with
realistic feature relationships and noise, so the model actually has
something to learn from.

NOTE: For a real project, replace this with a real dataset (e.g. the
Ames Housing dataset or a local market export) — synthetic data is a
stand-in so the pipeline below has something realistic to train on.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 800

square_feet = rng.normal(1800, 650, N).clip(400, 6000).round(0)
bedrooms = rng.integers(1, 6, N)
bathrooms = (rng.integers(2, 12, N) / 2).round(1)  # 1.0 - 5.5 in 0.5 steps
year_built = rng.integers(1950, 2026, N)
neighborhood_score = rng.uniform(1, 10, N).round(1)

house_age = 2026 - year_built

# Realistic-ish price formula with noise: bigger, newer, better-located
# houses cost more, with diminishing returns and some randomness.
base_price = (
    80 * square_feet
    + 12000 * bedrooms
    + 9000 * bathrooms
    - 400 * house_age
    + 15000 * neighborhood_score
    + 20000
)
noise = rng.normal(0, 25000, N)
price = (base_price + noise).clip(50000, None).round(0)

df = pd.DataFrame({
    "SquareFeet": square_feet.astype(int),
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "YearBuilt": year_built,
    "Neighborhood_Score": neighborhood_score,
    "Price": price.astype(int),
})

df.to_csv("house_data.csv", index=False)
print(f"Wrote {len(df)} rows to house_data.csv")
