import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import datetime
import os
import pickle

from src.rnn import RNN
from src import hyper_parameters as hp


def train_model(sess_file, rnn, data_handle, seed=None):
    """
    Run the train method of the rnn given in argument
    :param sess_file:
    :param rnn:
    :param data_handle:
    :param seed:
    :return:
    """
    print("Starting training...")
    mse = rnn.train(data_handle, sess_file=sess_file, seed=seed)
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("mse")
    ax.plot(mse["iteration"], mse["mse"])
    print("Training completed !!")


def prediction(sess_file, rnn, data_handle, test_date=None, nb_prediction=None):
    """
    Use the RNN given in argument to predict data.
    :param sess_file: tensorflow saved session file (e.g. "my_model/my_rnn")
    :param rnn: src.rnn object with hyper parameters matching hyper parameters from saved tf metagraph
    :param data_handle: DataHandler object with test_set for the prediction
    :param test_date: Date to predict. If absent the whole test_set will be predicted
    :param nb_prediction: Nb of predication requested.
    :return:
    """
    if test_date and not isinstance(test_date, datetime.datetime):
        raise TypeError("test_date shall be a datetime.datetime object. Got {}".format(type(test_date)))
    if not nb_prediction:
        nb_prediction = data_handle.test_set.shape[0]
    if test_date:
        input_start_date = test_date - datetime.timedelta(hours=rnn.nb_time_step)
        input_data, input_scaled = data_handle.get_sample(input_start_date, rnn.nb_time_step)
        test_set, test_scaled = data_handle.get_sample(test_date, nb_prediction)
    else:
        test_scaled = data_handle.test_scaled[rnn.nb_time_step:nb_prediction]
        test_set = data_handle.test_set[rnn.nb_time_step:nb_prediction]
        input_scaled = data_handle.test_scaled[:rnn.nb_time_step]
        input_data = data_handle.test_set[:rnn.nb_time_step]
    y_pred = rnn.run(sess_file=sess_file, input_set=input_scaled, test_set=test_scaled)
    mse_pred = np.mean(np.square(y_pred - test_scaled))
    y_pred = data_handle.inverse_transform(y_pred)
    return y_pred, test_set, mse_pred, input_data


def plot_result(y_pred, test_set, input_data):
    nb_pred = y_pred.shape[0]
    nb_time_step = input_data.shape[0]
    test_df = pd.DataFrame(test_set[:nb_pred].reshape(-1),
                           index=list(range(nb_time_step, nb_time_step + nb_pred)))
    input_df = pd.DataFrame(input_data.reshape(-1))
    df = pd.concat([input_df, test_df])
    pred_df = pd.DataFrame(y_pred.reshape(-1),
                           index=list(range(nb_time_step, nb_time_step + nb_pred)),
                           columns=["prediction"])
    df = pd.concat([df, pred_df], axis=1).set_index("time")
    print(df)
    ax = df.plot()
#    ax.text(1, 1, "coucou", fontsize=12, transform=ax.transAxes)
    plt.show()


def load_model_if_exists(sess_name, hyper_parameter, seed):
    if os.path.isfile(sess_name + ".param"):
        rnn = RNN(load_model=sess_name + ".param")
    else:
        return None
    check = hyper_parameter["learning_rate"] - rnn.learning_rate
    check += hyper_parameter["nb_input"] - rnn.nb_input
    check += hyper_parameter["nb_time_step"] - rnn.nb_time_step
    check += hyper_parameter["nb_neuron"] - rnn.nb_neuron
    check += hyper_parameter["batch_size"] - rnn.batch_size
    check += hyper_parameter["nb_iteration"] - rnn.nb_iteration
    if check == 0 and seed == rnn.seed:
        return rnn
    return None


def model_optimizer(data_handle, learning_rate, nb_neuron, nb_time_step, sess_folder=None):
    result = {"mse": [], "learning_rate": [], "nb_neuron": [], "nb_time_step": []}
    for lr in learning_rate:
        for num_neuron in nb_neuron:
            for num_time_step in nb_time_step:
                hyper_parameter = {
                    "learning_rate": lr,
                    "nb_input": hp.nb_input,
                    "nb_output": hp.nb_output,
                    "nb_time_step": num_time_step,
                    "nb_neuron": num_neuron,
                    "batch_size": hp.batch_size,
                    "nb_iteration": hp.nb_iteration
                }
                seed = 42
                print("----------------------------")
                print("lr={} ; nb_neuron={} ; num_time_step={}".format(lr, num_neuron, num_time_step))
                sess_name = sess_folder + "/" + "RNN_{}lr_{}inputs_{}neurons" \
                    .format(lr, num_time_step, num_neuron)
                rnn = load_model_if_exists(sess_name, hyper_parameter, seed)
                if not rnn:
                    rnn = RNN(hyper_parameter=hyper_parameter)
                    train_model(sess_name, rnn, data_handle, seed=seed)
                y_pred, test_set, mse_pred, input_data = prediction(sess_name, rnn, data_handle)
                print("mse prediction = ", mse_pred)
                result["mse"].append(mse_pred)
                result["learning_rate"].append(lr)
                result["nb_time_step"].append(num_time_step)
                result["nb_neuron"].append(num_neuron)
    result["mse_unscaled"] = data_handle.inverse_transform([x for x in result["mse"]].reshape(-1, 1))
    with open("opti_results", 'wb') as file:
        pickle.dump(result, file)
    print(result)
    #Axes3D.plot_wireframe(result["nb_time_step"], result["learning_rate"], result["mse"])
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(result["learning_rate"], result["mse"])
    x = [x for x, y in sorted(zip(result["nb_time_step"], result["mse"]))]
    y = [y for x, y in sorted(zip(result["nb_time_step"], result["mse"]))]
    ax2.plot(x, y)
    ax3.plot(result["nb_neuron"], result["mse"])
    plt.show()

