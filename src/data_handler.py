import sklearn.preprocessing as skl_pre
import datetime
import numpy as np


class DataHandler:

    def __init__(self, log_data):
        if "time" not in log_data.dtype.names:
            raise TypeError("log_data has no field named 'time'")
        if "count" not in log_data.dtype.names:
            raise TypeError("log_data has no field named 'count'")
        self.data = log_data.reshape(-1, 1)
        self.train_set = None
        self.test_set = None
        self.train_scaled = None
        self.test_scaled = None
        self.scale = None

    def train_test_split(self, split_date, test_set_first=False):
        """
        Split the data between a train set and a test set.
        Param:
            - data: data array to be split must contain a datetime index named "time"
            - split_date: data set is split as follow : [0:split_date-1] & [split_date:]
            - test_set_first=False -> if True the split is test [0:split_date-1]
        Return:
            train_set and test_set
        """
        if not isinstance(split_date, datetime.datetime):
            raise TypeError("The split_date shall be of type datetime.datetime. Got {}"
                            .format(type(split_date)))
        if split_date > self.data[-1]["time"] or split_date <= self.data[0]["time"]:
            raise ValueError("The split_date ({}) is date is not within the data set"
                             .format(split_date))
        for index, bucket in enumerate(self.data):
            if split_date < bucket["time"]:
                split_index = index
                break
        if not test_set_first:
            self.train_set = self.data[:split_index]
            self.test_set = self.data[split_index:]
        else:
            self.test_set = self.data[:split_index]
            self.train_set = self.data[split_index:]
        return self.train_set, self.test_set

    def scale_data(self):
        if self.train_set is None or self.test_set is None:
            raise AttributeError("Data must but split "
                                 "between test set and train set prior to being scaled")
        self.scale = skl_pre.MinMaxScaler()
        self.scale.fit(self.train_set["count"])
        self.train_scaled = self.scale.transform(self.train_set["count"])
        self.test_scaled = self.scale.transform(self.test_set["count"])
        return self.train_scaled, self.test_scaled

    def inverse_transform(self, data):
        if self.scale is None:
            raise AttributeError("No scale found. Please first create a scale using scale_data method")
        try:
            unscaled = self.scale.inverse_transform(data)
        except ValueError:
            unscaled = np.empty(shape=data.shape, dtype=np.float32)
            unscaled[:] = np.nan
        return unscaled

    def next_batch(self, batch_size, nb_time_step):
        """
        Return x and y batch of scaled data from the train set.
        :param batch_size: size of the batch
        :param nb_time_step: number of time step per batch
        :return: x_batch, y_batch with y_batch[i] = x_batch[i - 1]
        """
        if self.train_scaled is None:
            raise AttributeError("Data must but scaled prior to being batched")
        start = np.random.randint(len(self.train_scaled) - nb_time_step, size=batch_size)
        batch_x = np.zeros(shape=(batch_size, nb_time_step, self.train_scaled.shape[1]))
        batch_y = np.zeros(shape=(batch_size, nb_time_step, self.train_scaled.shape[1]))
        for i, s in enumerate(start):
            batch_x[i, :, :] = self.train_scaled[s: s + nb_time_step][:]
            batch_y[i, :, :] = self.train_scaled[s + 1: s + nb_time_step + 1][:]
        return batch_x.reshape(batch_size, nb_time_step, -1), batch_y.reshape(batch_size, nb_time_step, -1)

    def get_sample(self, start_date, sample_size):
        if not isinstance(start_date, datetime.datetime):
            raise TypeError("start_date and end_date shall be of type datetime.datetime.")
        index_start = -1
        for index, bucket in enumerate(self.data):
            if bucket["time"] >= start_date:
                index_start = index
                break
        index_end = index_start + sample_size
        if index_start < 0:
            raise ValueError("start_date is not compatible with data set")
        if index_end > self.data.shape[0]:
            raise ValueError("start_date '{}' is too close to end '{}' to get sample_size entry"
                             .format(start_date, self.data[-1]["time"]))
        sample = self.data[index_start:index_end]
        return sample, self.scale.transform(sample)


