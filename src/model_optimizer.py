import matplotlib.pyplot as plt
import pandas as pd
import datetime

from src.rnn import RNN
from src import hyper_parameters as hp


def train_model(sess_name, hyper_parameter, data_handle):
    rnn = RNN(hyper_parameter)
    print("Starting training...")
    mse = rnn.train(data_handle, sess_name=sess_name, seed=42)
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("mse")
    ax.plot(mse["iteration"], mse["mse"])
    print("Training completed !!")


def prediction(sess_name, rnn, data_handle, test_date, nb_prediction=24):
    input_start_date = test_date - datetime.timedelta(hours=hp.nb_time_step)
    input_data, input_scaled = data_handle.get_sample(input_start_date, hp.nb_time_step)
    test_set, test_scaled = data_handle.get_sample(test_date, nb_prediction)
    y_pred = rnn.run(sess_name=sess_name, input_set=input_scaled, test_set=test_scaled)
    y_pred = data_handle.inverse_transform(y_pred)
    return y_pred


def model_optimizer(data_handle, sess_name):
    nb_pred = y_pred.shape[0]
    test_df = pd.DataFrame(test_set[:nb_pred].reshape(-1),
                           index=list(range(hp.nb_time_step, hp.nb_time_step + nb_pred)))
    input_df = pd.DataFrame(input_data.reshape(-1))
    df = pd.concat([input_df, test_df])
    pred_df = pd.DataFrame(y_pred.reshape(-1),
                           index=list(range(hp.nb_time_step, hp.nb_time_step + nb_pred)),
                           columns=["prediction"])
    df = pd.concat([df, pred_df], axis=1).set_index("time")
    print(df)
    df.plot()
    plt.show()
