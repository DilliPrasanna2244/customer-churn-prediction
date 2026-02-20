import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load the CSV dataset into a pandas DataFrame.
    filepath: path to the CSV file
    """
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Fix data issues:
    - TotalCharges has spaces instead of numbers in some rows
    - Drop customerID (not useful for prediction)
    - Drop rows with nulls (very few)
    - Convert Churn from Yes/No to 1/0
    """
    # Fix TotalCharges — some values are empty strings " "
    # pd.to_numeric converts to number; errors='coerce' turns bad values to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop the customerID column — it's just a unique ID, not a feature
    df.drop('customerID', axis=1, inplace=True)

    # Drop rows with NaN values (only ~11 rows affected)
    df.dropna(inplace=True)

    # Convert target column: Yes → 1, No → 0
    # ML models need numbers, not strings
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print(f"✅ Data cleaned: {df.shape[0]} rows remaining")
    return df


def encode_features(df):
    """
    Convert all text columns to numbers using LabelEncoder.
    Example: 'Male'→1, 'Female'→0 | 'Yes'→1, 'No'→0
    """
    le = LabelEncoder()

    # Select only columns with text (object dtype)
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    print(f"✅ Encoded {len(cat_cols)} categorical columns")
    return df


def split_and_scale(df):
    """
    1. Separate features (X) and target (y)
    2. Split into 80% train, 20% test
    3. Scale features so all values are on same range
    """
    X = df.drop('Churn', axis=1)   # everything except Churn
    y = df['Churn']                 # only Churn column

    # 80% for training, 20% for testing
    # random_state=42 means same split every time you run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # StandardScaler: converts values so mean=0 and std=1
    # fit_transform on train, only transform on test (avoid data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()