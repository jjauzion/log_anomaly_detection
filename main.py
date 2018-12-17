import datetime
import matplotlib.pyplot as plt
import pandas as pd

from src import log_handler as lg
from src import data_handler as dh
from src.rnn import RNN
from src import hyper_parameters as hp
from src import model

"""
#JOIN LOG DATA
log1 = lg.Log()
log1.load("MNJ_log_from_2018-05-16_to_2018-06-23.npy")
log2 = lg.Log()
log2.load("MNJ_log_from_2018-09-04_to_2018-10-19.npy")
log = lg.Log()
log.concat(log1, log2)
log.save(name="MNJ_16mai-23jui--04sep-19oct.npy")
exit()
"""

"""
#IMPORT FROM KIBANA
log = lg.Log()
date_from = "2018-09-04"
date_to = "2018-10-19"
query = {
    "size": 0,
    "query": {
        "range": {
            "@timestamp": {
                "gte": date_from,
                "lte": date_to,
                "format": "yyyy-MM-dd"
            }
        }
    },
    "aggs": {
        "byhour": {
            "date_histogram": {
                "field": "@timestamp",
                "interval": "hour"
            }
        }
    }
}
log_file_name = "test.npy"
date_from = datetime.datetime.strptime(date_from, "%Y-%m-%d")
date_to = datetime.datetime.strptime(date_to, "%Y-%m-%d")
log.import_data(index="varnishlogbeat", query=query)
#log.save(name=log_file_name)
log.plot()
plt.show()
exit()
"""

#LOAD LOG DATA FROM DISK
log_file_name = "MNJ_16mai-23jui--04sep-19oct.npy"
log = lg.Log()
log.load(log_file_name)
log.plot()

#TRAIN TEST SPLIT
split_date = datetime.datetime.strptime("01-07-2018", "%d-%m-%Y")
data_handle = dh.DataHandler(log_data=log.data)
data_handle.train_test_split(split_date, test_set_first=True)
data_handle.scale_data()

"""
#MODEL OPTIMIZATION
result = model.model_optimizer(data_handle, learning_rate=[0.1, 0.2], nb_neuron=[10, 20],
                      nb_time_step=[12, 24], activation_fct=["tanh", "relu"],
                      sess_folder="model", seed=118)
model.plot_optimization_result(result=result)
"""

"""
#CREATING MODEL
hyper_parameter = {
    "learning_rate": 0.01,
    "nb_input": hp.nb_input,
    "nb_output": hp.nb_output,
    "nb_time_step": 48,
    "nb_neuron": 50,
    "batch_size": hp.batch_size,
    "nb_iteration": hp.nb_iteration,
    "activation_fct": model.get_activation_fct("tanh")
}
rnn = RNN(hyper_parameter, report_iter_freq=100)
"""

"""
#SESSION NAME
sess_file = "model/RNN_{}lr_{}inputs_{}neurons".format(hyper_parameter["learning_rate"],
                                                       hyper_parameter["nb_time_step"],
                                                       hyper_parameter["nb_neuron"])
"""

"""
#TRAIN
model.train_model(sess_file, rnn, data_handle, seed=42, plot=True, evaluate=True)
"""
sess_file = "model/RNN_0.01lr_48inputs_50neurons_10000iter"
rnn = RNN(load_model=sess_file + ".param")
mse_vs_iter = pd.DataFrame(rnn.mse)
mse_vs_iter = mse_vs_iter.set_index("iteration")
print(mse_vs_iter)
mse_vs_iter.plot()
plt.show()


"""
#PREDICTION
#sess_file = "model/RNN_0.03lr_122inputs_100neurons_actFct-tanh"
#rnn = RNN(load_model=sess_file + ".param")
y_pred, test_set, mse, input_data = model.prediction(sess_file, rnn, data_handle)
model.plot_prediction(y_pred, test_set, input_data)
plt.show()
"""

"""
sess_file = "model/RNN_0.03lr_122inputs_100neurons_actFct-tanh"
model.eval(sess_file, data_handle, 42, 10)
plt.show()
"""

"""
#PREDICTION BLIND
nb_pred = 16
test_date = datetime.datetime.strptime("2018-10-19_06", "%Y-%m-%d_%H")
input_start_date = test_date - datetime.timedelta(hours=hp.nb_time_step)
input_data, input_scaled = data_handle.get_sample(input_start_date, hp.nb_time_step)
test_set, test_scaled = data_handle.get_sample(test_date, nb_pred)
y_pred = rnn.run(sess_name=sess_name, input_set=input_scaled, nb_pred=nb_pred)
y_pred = data_handle.inverse_transform(y_pred)
"""

