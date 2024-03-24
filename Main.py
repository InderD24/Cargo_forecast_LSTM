import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DataPreprocess import DataPreprocessor
from ModelEval import LSTMModel

def plot_results_with_future(df, pred_dates, predicted_TEU_inv, actual_TEU_inv, original_TEU, cleaned_TEU, future_dates, future_TEU_predictions):
    #looking at 2023 and 2024 only for simplicity 
    plt.figure(figsize=(15, 7))
    filter_mask = (df.index.year == 2023) | (df.index.year == 2024)

    # Original TEU volumes with outliers
    plt.plot(df[filter_mask].index, original_TEU[filter_mask],
             label='Original Actual TEU (with outliers)', color='orange', marker='o', linestyle='--')

    # Cleaned TEU volumes (outliers are mean of themselves)
    plt.plot(df[filter_mask].index, cleaned_TEU[filter_mask],
             label='Cleaned Actual TEU', color='blue', marker='o')

    # Predicted TEU volumes 
    predicted_mask = (pred_dates.year == 2023) | (pred_dates.year == 2024)
    plt.plot(pred_dates[predicted_mask], predicted_TEU_inv[predicted_mask],
             label='Predicted TEU', color='red', marker='x')

    plt.plot(future_dates, future_TEU_predictions, label='Future Predicted TEU', color='green', marker='x', linestyle='--')
    plt.title('TEU Volumes for 2023-2024 and Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('TEU Volume')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

def predict_future_steps(model, last_sequence, scaler, n_steps=10):
    future_predictions_scaled = model.predict_future(
        last_sequence, steps=n_steps)
    future_predictions = scaler.inverse_transform(
        future_predictions_scaled.reshape(-1, 1))

    return future_predictions.flatten()


def main():
    file_path = 'Put your file path here for the TEUcount'
    n_steps_in, n_steps_out = 3, 1
    target_column = 'TEU'

    preprocessor = DataPreprocessor(
        file_path, n_steps_in, n_steps_out, target_column)
    preprocessor.load_data()
    original_TEU = preprocessor.df[target_column].copy()
    preprocessor.clean_data()  #
    preprocessor.scale_data()  
    cleaned_TEU = preprocessor.df[target_column].copy()

    X, y = preprocessor.create_sequences()  
    X = X.reshape((X.shape[0], X.shape[1], 1))  

    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:
                                         split], X[split:], y[:split], y[split:]

    model = LSTMModel(n_steps_in, n_steps_out)  
    model.train(X_train, y_train)  
    model.evaluate(X_test, y_test)

    predicted_TEU = model.predict(X_test)
    predicted_TEU_inv = preprocessor.scaler.inverse_transform(
        predicted_TEU)

    actual_TEU_inv = preprocessor.scaler.inverse_transform(
        y_test.reshape(-1, 1))

    pred_dates = preprocessor.df.index[-len(predicted_TEU_inv):]

    last_known_sequence = X_test[-1].reshape((1, n_steps_in, 1))
    n_future_steps = 30  
    future_TEU_predictions = predict_future_steps(
        model, last_known_sequence, preprocessor.scaler, n_steps=n_future_steps)  

    last_date = preprocessor.df.index[-1] 
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=n_future_steps, freq='D')  

    plot_results_with_future(df=preprocessor.df, pred_dates=pred_dates, predicted_TEU_inv=predicted_TEU_inv,
                             actual_TEU_inv=actual_TEU_inv, original_TEU=original_TEU,
                             cleaned_TEU=cleaned_TEU, future_dates=future_dates,
                             future_TEU_predictions=future_TEU_predictions)

if __name__ == '__main__':
    main()
