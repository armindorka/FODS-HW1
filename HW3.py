from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

"""
HW.py

This file contains the skeleton for the functions you need to implement as part of your homework.
Each function corresponds to a specific task and includes instructions on what is expected.
"""

# Task 2: Data Cleaning
def clean_data(df):
    df_imputed = df.copy()  # Create a copy to avoid modifying the original DataFrame

    # List of columns expected to be numeric (convertible to numbers)
    numeric_col = [
        "Age", "Sex", "ChestPainType", "RestBP", "Chol", "FBS",
        "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal", "Num"
    ]

    # Convert specified columns to numeric, coercing invalid values to NaN
    for col in numeric_col:
        if col in df_imputed.columns:
            df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

    # Define valid bounds for each numeric column (used for outlier removal)
    bounds = {
        "Age": (18, 100),
        "Sex": (0, 1),
        "ChestPainType": (1, 4),
        "RestBP": (0, 200),
        "Chol": (0, 520),
        "FBS": (0, 1),
        "RestECG": (0, 2),
        "MaxHR": (30, 250),
        "ExAng": (0, 1),
        "Oldpeak": (0, 10),
        "Slope": (1, 3),
        "Ca": (0, 3),
        "Thal": (3, 7),
        "Num": (0, 4)  
    }

    # Replace values outside valid ranges with NaN
    for col, (low, high) in bounds.items():
        if col in df_imputed.columns:
            df_imputed.loc[(df_imputed[col] < low) | (df_imputed[col] > high), col] = np.nan

    # Count missing values after cleaning
    NaN_count = df_imputed.isnull().sum().to_dict()

    return df_imputed, NaN_count


# Task 3: Categorical Features Imputation
def impute_missing_categorical(df_train, df_val, df_test, categorical_columns):
    """
    Task: Categorical Features Imputation
    --------------------------------------
    Handles missing categorical features using KNN imputation.
    """

    # If no categorical columns, return empty DataFrames matching index
    if not categorical_columns:
        return (
            pd.DataFrame(index=df_train.index),
            pd.DataFrame(index=df_val.index),
            pd.DataFrame(index=df_test.index),
        )

    # Extract categorical subsets from each dataset
    train_cat = df_train[categorical_columns].copy()
    val_cat = df_val[categorical_columns].copy()
    test_cat = df_test[categorical_columns].copy()

    # Create encoded copies for numerical imputation
    encoded_train = train_cat.copy()
    encoded_val = val_cat.copy()
    encoded_test = test_cat.copy()

    column_to_codebook = {}  # Maps category to numeric code
    column_to_values = {}    # Stores mappings for decoding later

    # Encode each categorical column numerically
    for col in categorical_columns:
        cat = pd.Categorical(train_cat[col])
        categories = [c for c in cat.categories]
        value_to_code = {val: idx for idx, val in enumerate(categories)}
        column_to_codebook[col] = value_to_code
        original_codes = np.array(list(value_to_code.values()), dtype=float)
        column_to_values[col] = {
            "value_to_code": value_to_code,
            "code_to_value": {code: val for val, code in value_to_code.items()},
            "original_codes": original_codes,
        }

        # Map string categories to numeric codes
        def map_to_code(x):
            if pd.isna(x):
                return np.nan
            return value_to_code.get(x, np.nan)

        encoded_train[col] = train_cat[col].map(map_to_code)
        encoded_val[col] = val_cat[col].map(map_to_code)
        encoded_test[col] = test_cat[col].map(map_to_code)

    # Initialize KNN imputer (k=5, distance-weighted)
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputer.fit(encoded_train.values)

    train_imputed = imputer.transform(encoded_train.values)
    val_imputed = imputer.transform(encoded_val.values)
    test_imputed = imputer.transform(encoded_test.values)

    # Helper: round imputed codes to nearest valid category code
    def round_col(imputed_codes_matrix, reference_index):
        rounded = np.empty_like(imputed_codes_matrix)
        for j, col in enumerate(categorical_columns):
            col_vals = imputed_codes_matrix[:, j]
            original_codes = column_to_values[col]["original_codes"]
            mask = ~np.isnan(col_vals)
            new_col = col_vals.copy()
            if np.any(mask):
                nearest = []
                for v in new_col[mask]:
                    distances = np.abs(original_codes - v)
                    min_dist = np.min(distances)
                    candidates = original_codes[distances == min_dist]
                    chosen = np.min(candidates) if candidates.size > 0 else np.nan
                    nearest.append(chosen)
                new_col[mask] = np.array(nearest)
            rounded[:, j] = new_col
        # Decode numeric codes back to categorical labels
        decoded = pd.DataFrame(rounded, index=reference_index, columns=categorical_columns)
        for j, col in enumerate(categorical_columns):
            code_to_value = column_to_values[col]["code_to_value"]
            decoded[col] = decoded[col].apply(lambda x: code_to_value.get(int(x), np.nan) if pd.notna(x) else np.nan)
        return decoded

    # Decode imputed matrices back to DataFrames
    train_cat_imputed = round_col(train_imputed, train_cat.index)
    val_cat_imputed = round_col(val_imputed, val_cat.index)
    test_cat_imputed = round_col(test_imputed, test_cat.index)

    return train_cat_imputed, val_cat_imputed, test_cat_imputed


# Task 4: Numerical Features Imputation
def impute_numerical_features(df_train, df_val, df_test, numerical_columns):
    # Handle edge case where no numerical columns are specified
    if not numerical_columns:
        return (
            pd.DataFrame(index=df_train.index),
            pd.DataFrame(index=df_val.index),
            pd.DataFrame(index=df_test.index),
        )

    # Create copies for safety
    train_numeric = df_train[numerical_columns].copy()
    val_numeric = df_val[numerical_columns].copy()
    test_numeric = df_test[numerical_columns].copy()

    # Helper to find columns that already have no missing values
    def get_completed_columns():
        cols = []
        for col in numerical_columns:
            if not (train_numeric[col].isna().any() or val_numeric[col].isna().any() or test_numeric[col].isna().any()):
                cols.append(col)
        return set(cols)

    completed_cols = get_completed_columns()

    # Iteratively impute missing values for each numeric feature
    while True:
        # Build summary of missing values per column
        missing_summary = {}
        for col in numerical_columns:
            total_missing = train_numeric[col].isna().sum() + val_numeric[col].isna().sum() + test_numeric[col].isna().sum()
            if total_missing > 0:
                missing_summary[col] = total_missing
        if not missing_summary:
            break  # Exit loop if no missing values remain

        # Choose target column with fewest missing values
        target_numeric = min(missing_summary, key=missing_summary.get)
        available_features = [c for c in numerical_columns if c != target_numeric and c in completed_cols]

        # If no features to use for prediction, impute using mean
        if len(available_features) == 0:
            mean_val = train_numeric[target_numeric].mean()
            for dataset in (train_numeric, val_numeric, test_numeric):
                mask_missing = dataset[target_numeric].isna()
                if mask_missing.any():
                    dataset.loc[mask_missing, target_numeric] = mean_val
            completed_cols.add(target_numeric)
            continue

        # Prepare training data for LASSO regression
        mask_not_missing = train_numeric[target_numeric].notna()
        X_train = train_numeric.loc[mask_not_missing, available_features]
        y_train = train_numeric.loc[mask_not_missing, target_numeric]

        # Drop any incomplete rows from training
        mask_complete = ~X_train.isna().any(axis=1)
        X_train = X_train.loc[mask_complete]
        y_train = y_train.loc[mask_complete]

        # If no complete training data, fallback to mean imputation
        if len(y_train) == 0:
            mean_val = train_numeric[target_numeric].mean()
            for dataset in (train_numeric, val_numeric, test_numeric):
                mask_missing = dataset[target_numeric].isna()
                dataset.loc[mask_missing, target_numeric] = mean_val
            completed_cols.add(target_numeric)
            continue

        # Train LASSO regression to predict missing numeric values
        lasso_model = Lasso(alpha=1.0, max_iter=10000)
        lasso_model.fit(X_train.values, y_train.values)

        # Helper function to fill missing entries using LASSO predictions
        def fill_missing(dataset):
            mask_missing = dataset[target_numeric].isna()
            if not mask_missing.any():
                return
            X_missing = dataset.loc[mask_missing, available_features]
            mask_complete_rows = ~X_missing.isna().any(axis=1)
            if mask_complete_rows.any():
                preds = lasso_model.predict(X_missing.loc[mask_complete_rows].values)
                dataset.loc[mask_missing[mask_missing].index[mask_complete_rows], target_numeric] = preds

        # Apply model-based imputation
        for dataset in (train_numeric, val_numeric, test_numeric):
            fill_missing(dataset)

        # Replace any remaining NaNs with column mean
        mean_val = train_numeric[target_numeric].mean()
        for dataset in (train_numeric, val_numeric, test_numeric):
            mask_missing = dataset[target_numeric].isna()
            dataset.loc[mask_missing, target_numeric] = mean_val

        completed_cols.add(target_numeric)

    # Return imputed datasets
    train_imputed_lasso = train_numeric
    val_imputed_lasso = val_numeric
    test_imputed_lasso = test_numeric

    return train_imputed_lasso, val_imputed_lasso, test_imputed_lasso


# Task: Merge imputed categorical and numerical DataFrames
def merge_imputed(df_cat, df_num):
    """
    Simply merges categorical and numerical imputed DataFrames on their index.
    """
    merged = pd.concat([df_cat, df_num], axis=1)
    return merged


# Task 5: Classification Using a Single Split
def train_and_evaluate_single_split(X_train, X_val, y_train, y_val, cat_cols, num_cols, model, hp):
    """
    Builds and evaluates a classification pipeline using a single train/validation split.
    """

    # Auto-detect categorical and numerical columns if not provided
    if cat_cols is None:
        categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    else:
        categorical_cols = list(cat_cols)
    if num_cols is None:
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerical_cols = list(num_cols)

    # Define preprocessing steps: OneHotEncode cats, StandardScale nums
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )

    # Combine preprocessing and model into a single pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Apply provided hyperparameters (if any)
    if hp:
        pipeline.set_params(**{f"model__{k}": v for k, v in hp.items()})

    # Train and evaluate model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    score = f1_score(y_val, y_pred)

    return {"params": hp, "F1 scores": score}


# Task 6: Classification Using Cross-Validation
def train_and_evaluate_cross_validation(X, y, model, cat_cols, num_cols, hp, cv):
    # Use StratifiedKFold to maintain class balance across folds
    strat_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=8)

    # Detect categorical and numerical columns if not explicitly given
    if cat_cols is None:
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    else:
        categorical_cols = list(cat_cols)

    if num_cols is None:
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerical_cols = list(num_cols)

    eval_scores = []  # Store F1 scores from each fold

    # Iterate through cross-validation folds
    for train_idx, val_idx in strat_fold.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Define preprocessing and modeling pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ("num", StandardScaler(), numerical_cols),
            ]
        )

        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

        # Apply hyperparameters if specified
        if hp:
            pipeline.set_params(**{f"model__{k}": v for k, v in hp.items()})

        # Fit model and evaluate on validation fold
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        eval_scores.append(f1_score(y_val, y_pred))

    # Compute average F1 score across folds
    avg_f1 = float(np.mean(eval_scores)) if len(eval_scores) > 0 else np.nan
    return {"params": hp, "Average F1 scores": avg_f1}
