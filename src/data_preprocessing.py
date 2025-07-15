import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path, sep=';')
    return df

def preprocess_data(df):
    df['dropout'] = df['G3'].apply(lambda x: 1 if x < 10 else 0)
    df = df.drop(['G3'], axis=1)

    le_dict = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('dropout', axis=1)
    y = df['dropout']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_dict, X.columns.tolist()
