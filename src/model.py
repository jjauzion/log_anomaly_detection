import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import datetime
import os
import pickle
import tensorflow as tf

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


def plot_prediction(y_pred, test_set, input_data):
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
    df["diff"] = abs(df["prediction"] - df["count"])
    print(df)
    df.plot()
    """
    fig, ax1 = plt.subplots()
    ax1.plot(df.index, df["count"], label="count")
    ax1.plot(df.index, df["prediction"], label="prediction")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["diff"], '--r', label="delta")
    ax2.set_ylim(ax1.get_ylim())
#    ax.text(1, 1, "coucou", fontsize=12, transform=ax.transAxes)
    ax2.legend(loc="center right")
    """
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
    if check == 0 and seed == rnn.seed and hyper_parameter["activation_fct"] == rnn.activation_fct:
        return rnn
    return None


def get_activation_fct(fct_name):
    if fct_name == "tanh":
        return tf.nn.tanh
    elif fct_name == "relu":
        return tf.nn.relu
    else:
        raise ValueError("{} is not recognized as an activation function.".format(fct_name))


def model_optimizer(data_handle, learning_rate, nb_neuron, nb_time_step, activation_fct,
                    sess_folder=None, seed=None):
    length = len(learning_rate) * len(nb_neuron) * len(nb_time_step) * len(activation_fct)
    result = np.zeros(length, dtype=[('mse_training', 'f8'),
                                     ('mse_test', 'f8'),
                                     ('learning_rate', 'f8'),
                                     ('nb_neuron', 'i8'),
                                     ('nb_time_step', 'i8'),
                                     ('activation_fct', np.unicode, 16)])
    i = 0
    for lr in learning_rate:
        for num_neuron in nb_neuron:
            for num_time_step in nb_time_step:
                for activation in activation_fct:
                    hyper_parameter = {
                        "learning_rate": lr,
                        "nb_input": hp.nb_input,
                        "nb_output": hp.nb_output,
                        "nb_time_step": num_time_step,
                        "nb_neuron": num_neuron,
                        "batch_size": hp.batch_size,
                        "nb_iteration": hp.nb_iteration,
                        "activation_fct": get_activation_fct(activation)
                    }
                    print("----------------------------")
                    print("lr={} ; nb_neuron={} ; num_time_step={} ; actFct={}"
                          .format(lr, num_neuron, num_time_step, activation))
                    sess_name = sess_folder + "/" + "RNN_{}lr_{}inputs_{}neurons_actFct-{}" \
                        .format(lr, num_time_step, num_neuron, activation)
                    rnn = load_model_if_exists(sess_name, hyper_parameter, seed)
                    if not rnn:
                        rnn = RNN(hyper_parameter=hyper_parameter)
                        train_model(sess_name, rnn, data_handle, seed=seed)
                    y_pred, test_set, mse_pred, input_data = prediction(sess_name, rnn, data_handle)
                    print("mse prediction = ", mse_pred)
                    result["mse_training"][i] = rnn.mse_training
                    result["mse_test"][i] = mse_pred
                    result["nb_time_step"][i] = num_time_step
                    result["nb_neuron"][i] = num_neuron
                    result["learning_rate"][i] = lr
                    result["activation_fct"][i] = activation
                    i += 1
    with open("opti_results", 'wb') as file:
        pickle.dump(result, file)
    return result


def plot_optimization_result(result=None, file=None):
    if (result is not None and file is not None) or (result is None and file is None):
        raise AttributeError("One and only one of 'result' and 'file' shall be defined.")
    if file:
        with open(file, 'rb') as res_file:
            result = pickle.load(res_file)
    result_df = pd.DataFrame(result)
    result_df.sort_values("mse_test", inplace=True)
    print(result_df)
    """
    #Axes3D.plot_wireframe(result["nb_time_step"], result["learning_rate"], result["mse"])
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(result["learning_rate"], result["mse_test"])
    x = [x for x, y in sorted(zip(result["nb_time_step"], result["mse_test"]))]
    y = [y for x, y in sorted(zip(result["nb_time_step"], result["mse_test"]))]
    ax2.plot(x, y)
    ax3.plot(result["nb_neuron"], result["mse_test"])
    plt.show()
    """
    return result_df


