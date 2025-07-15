from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_and_evaluate

def main():
    df = load_data("data/student-por.csv")
    X, y, scaler, le_dict, feature_names = preprocess_data(df)
    train_and_evaluate(X, y, scaler, le_dict, feature_names)

if __name__ == "__main__":
    main()
