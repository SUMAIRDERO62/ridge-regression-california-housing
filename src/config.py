"""Project-wide constants for Task 2."""

RANDOM_STATE = 42
TARGET_NAME = "MedHouseVal"          # median house value, in units of $100,000
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Categorical feature engineered from HouseAge (the raw dataset is all-numeric)
AGE_BINS = [0, 15, 30, 45, 100]
AGE_LABELS = ["new", "mid", "old", "very_old"]

# Columns where missing values are injected to exercise the imputer
MISSING_COLUMNS = ["AveRooms", "AveBedrms", "Population"]
