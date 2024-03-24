from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.regularizers import l1_l2

class LSTMModel:
    def __init__(self, n_steps_in, n_steps_out):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.model = self.build_model()

    #simpler
    def build_model(self):
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True,
                 input_shape=(self.n_steps_in, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dense(self.n_steps_out)

        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    #complex model trying to get more accurate future predictions
    #def build_model(self):
        #model = Sequential([
         #   LSTM(50, activation='relu', return_sequences=True,
          #       input_shape=(self.n_steps_in, 1),
           #      dropout=0.1,  
            #     recurrent_dropout=0.1, 
             #    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), 
              #   recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),  
               #  bias_regularizer=l1_l2(l1=1e-5, l2=1e-4)), 
            #LSTM(50, activation='relu', return_sequences=True,
            #     dropout=0.1,  
            #     recurrent_dropout=0.1), 
            #Dropout(0.2),
            #LSTM(50, activation='relu'),
            #Dense(self.n_steps_out)
        #])
        #model.compile(optimizer='adam', loss='mean_squared_error')
        #return model

    def train(self, X_train, y_train):
        es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        self.model.fit(X_train, y_train, epochs=100,
                       validation_split=0.2, verbose=1, callbacks=[es])

    def evaluate(self, X_test, y_test):
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Loss: {test_loss}')
        return test_loss

    def predict(self, X):
        return self.model.predict(X)

    def predict_future(self, input_sequence, steps =10):
        future_predictions =[]
        current_sequence = input_sequence
        
        for _ in range(steps):
            next_step_prediction = self.model.predict(current_sequence)
            next_step_prediction = next_step_prediction.reshape(1, 1, -1)
            current_sequence = np.append(current_sequence[:, 1:, :], next_step_prediction, axis=1)
            future_predictions.append(next_step_prediction.flatten()[0]) 
        return np.array(future_predictions)
    

