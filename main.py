import datetime
import matplotlib.pyplot as plt

from src import log_handler as lg
from src import data_handler as dh
from src.rnn import RNN
from src import hyper_parameters as hp

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


#SPECIFIC QUERY
"""
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
                "interval": "minute"
            }
        }
    }
}
"""

"""
#IMPORT FROM KIBANA
log = lg.Log()
date_from = "2018-05-16"
date_to = "2018-06-23"
save_name = "MNJ_log_from_{}_to_{}.npy".format(date_from, date_to)
date_from = datetime.datetime.strptime(date_from, "%Y-%m-%d")
date_to = datetime.datetime.strptime(date_to, "%Y-%m-%d")
log.import_data(backend_tag_keyword="viacom-mnj", gte=date_from, lte=date_to)
log.save(name=save_name)
log.plot()
plt.show()
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

#CREATING MODEL
rnn = RNN(hp.learning_rate, hp.nb_input, hp.nb_output,
          hp.nb_time_step, hp.nb_neuron, hp.batch_size, hp.nb_iteration)

#SESSION NAME
sess_name = "model/RNN_test_seed42"

#TRAIN RNN
print("Starting training...")
mse = rnn.train(data_handle, save_name=sess_name, seed=42)
fig, ax = plt.subplots()
ax.set_xlabel("iteration")
ax.set_ylabel("mse")
ax.plot(mse["iteration"], mse["mse"])
print("Training completed !!")

#PREDICTION WITH TEST_SET
test_date = datetime.datetime.strptime("2018-06-01", "%Y-%m-%d")
input_start_date = test_date - datetime.timedelta(hours=hp.nb_time_step)
input_data, input_scaled = data_handle.get_sample(input_start_date, hp.nb_time_step)
test_set, test_scaled = data_handle.get_sample(test_date, 24)
y_pred = rnn.run(sess_name=sess_name, input_set=input_scaled, test_set=test_scaled)
y_pred = data_handle.inverse_transform(y_pred)

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

#PLOT RESULT
import pandas as pd
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
