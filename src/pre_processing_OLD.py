import sklearn.preprocessing as skl_pre
import datetime
import numpy as np


def train_test_split(data, split_date):
    """
    Split the data between a train set and a test set
    Param:
        - data: data array to be split must contain a datetime index named "time"
        - split_date: before split date -> train data / after split date -> test set
    Return:
        train_set and test_set
    """

    data = data.reshape(-1, 1)
    if not isinstance(split_date, datetime.datetime):
        raise TypeError("The split_date shall be of type datetime.datetime. Got {}"
                        .format(type(split_date)))
    if split_date > data[-1]["time"] or split_date <= data[0]["time"]:
        raise ValueError("The split_date ({}) is date is not within the data set"
                         .format(split_date))
    for index, bucket in enumerate(reversed(data)):
        if split_date >= bucket["time"]:
            split_index = index
            break
    train_set = data[:-split_index]
    test_set = data[-split_index:]
    return train_set, test_set


def scale_data(train_set, test_set):
    scaler = skl_pre.MinMaxScaler()
    scaler.fit(train_set)
    return scaler.transform(train_set), scaler.transform(test_set), scaler


def next_batch(train_set, batch_size, nb_time_step):
    start = np.random.randint(len(train_set) - nb_time_step, size=batch_size)
    batch_x = np.zeros(shape=(batch_size, nb_time_step, train_set.shape[1]))
    batch_y = np.zeros(shape=(batch_size, nb_time_step, train_set.shape[1]))
    for i, s in enumerate(start):
        batch_x[i, :, :] = train_set[s: s + nb_time_step][:]
        batch_y[i, :, :] = train_set[s + 1: s + nb_time_step + 1][:]
    return batch_x.reshape(batch_size, nb_time_step, -1), batch_y.reshape(batch_size, nb_time_step, -1)

