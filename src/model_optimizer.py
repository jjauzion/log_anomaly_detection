import matplotlib.pyplot as plt
import pandas as pd
import datetime

from src import log_handler as lg
from src import data_handler as dh
from src.rnn import RNN
from src import hyper_parameters as hp


def model_optimizer(data_handle, sess_name):
    rnn = RNN(hp.learning_rate, hp.nb_input, hp.nb_output,
              hp.nb_time_step, hp.nb_neuron, hp.batch_size, hp.nb_iteration)
    test_date = datetime.datetime.strptime("2018-06-01", "%Y-%m-%d")
    input_start_date = test_date - datetime.timedelta(hours=hp.nb_time_step)
    input_data, input_scaled = data_handle.get_sample(input_start_date, hp.nb_time_step)
    test_set, test_scaled = data_handle.get_sample(test_date, 24)
    y_pred = rnn.run(sess_name=sess_name, input_set=input_scaled, test_set=test_scaled)
    y_pred = data_handle.inverse_transform(y_pred)
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
