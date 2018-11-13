import datetime
import matplotlib.pyplot as plt

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
save_name = "test.npy"
date_from = datetime.datetime.strptime(date_from, "%Y-%m-%d")
date_to = datetime.datetime.strptime(date_to, "%Y-%m-%d")
log.import_data(index="varnishlogbeat", query=query)
#log.save(name=save_name)
log.plot()
plt.show()
exit()
"""

#LOAD LOG DATA FROM DISK
save_name = "MNJ_16mai-23jui--04sep-19oct.npy"
log = lg.Log()
log.load(save_name)
log.plot()

#TRAIN TEST SPLIT
split_date = datetime.datetime.strptime("01-07-2018", "%d-%m-%Y")
data_handle = dh.DataHandler(log_data=log.data)
data_handle.train_test_split(split_date, test_set_first=True)
data_handle.scale_data()

#MODEL OPTIMIZATION
model.model_optimizer(data_handle, learning_rate=[0.001],
                      nb_neuron=[200], nb_time_step=[48, 96],
                      sess_folder="model")

"""
#CREATING MODEL
hyper_parameter = {
    "learning_rate": 0.01,
    "nb_input": hp.nb_input,
    "nb_output": hp.nb_output,
    "nb_time_step": 120,
    "nb_neuron": 200,
    "batch_size": hp.batch_size,
    "nb_iteration": hp.nb_iteration
}
rnn = RNN(hyper_parameter)

#SESSION NAME
sess_file = "model/RNN_{}lr_{}inputs_{}neurons".format(hyper_parameter["learning_rate"],
                                                       hyper_parameter["nb_time_step"],
                                                       hyper_parameter["nb_neuron"])

#TRAIN with model.py
model.train_model(sess_file, rnn, data_handle, seed=42)

#PREDICTION with model.py
test_date = datetime.datetime.strptime("2018-10-19", "%Y-%m-%d")
y_pred, test_set, mse, input_data = \
    model.prediction(sess_file, rnn, data_handle, test_date=test_date, nb_prediction=24)
model.plot_result(y_pred, test_set, input_data)
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

