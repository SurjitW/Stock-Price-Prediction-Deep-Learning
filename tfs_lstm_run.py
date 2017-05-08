from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tfs_lstm, time #helper libraries
import tfs_download_symbol_prices
import os
import sys
import tfs_config as c


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def main():
    #Step 1 Load Data
    d = tfs_download_symbol_prices.DownloadSymbolPrices()
    data = d.download_data();
    X_train, y_train, X_test, y_test = tfs_lstm.set_data(data, 50, True)
    #X_train, y_train, X_test, y_test = tfs_lstm.load_data(c.symbol, sys.path[0] + '\\data\\sp500.csv', 50, True)
    #Step 2 Build Model
    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time : ', time.time() - start)

    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=1,
        validation_split=0.05)

    #Step 4 - Plot the predictions!
    predictions = tfs_lstm.predict_sequences_multiple(model, X_test, 50, 50)
    tfs_lstm.plot_results_multiple(c.symbol, predictions, y_test, 50)

if __name__ == "__main__":
    main()
