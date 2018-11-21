import tensorflow as tf
import numpy as np
import pickle


class RNN:

    def __init__(self, hyper_parameter=None, load_model=None, report_iter_freq=100):
        """
        Create a RNN object either from a saved model or from the hyper parameters in argument.
        :param hyper_parameter: Create a RNN object from the hyper parameter.
        :param load_model: Create a RNN object from a saved file (*.param file).
        :param report_iter_freq: Report iteration frequency during training (every 100 iteration by default).
        """
        if (not hyper_parameter and not load_model) or (hyper_parameter and load_model):
            raise AttributeError("One and only one of "
                                 "hyper_parameter and load_model shall be defined.")
        self._nan = False
        if load_model:
            self._load_parameter(load_model)
        else:
            self.learning_rate = hyper_parameter["learning_rate"]
            self.nb_input = hyper_parameter["nb_input"]
            self.nb_output = hyper_parameter["nb_output"]
            self.nb_time_step = hyper_parameter["nb_time_step"]
            self.nb_neuron = hyper_parameter["nb_neuron"]
            self.batch_size = hyper_parameter["batch_size"]
            self.nb_iteration = hyper_parameter["nb_iteration"]
            self.activation_fct = hyper_parameter["activation_fct"]
            if type(self.activation_fct) != type(tf.nn.tanh):
                raise TypeError("hyper_parameters['activation_fct'] shall be of type '{}'".format(type(tf.nn.tanh)))
            self.report_iter_freq = report_iter_freq
            self.seed = None
            self.avg_mse_training = None
            self.mse = np.zeros(shape=self.nb_iteration // self.report_iter_freq, dtype=[
                ('iteration', 'i8'), ('mse_batch', 'f8'), ('mse_training', 'f8'), ('mse_test', 'f8')])
            self.mse["iteration"] = np.arange(
                self.report_iter_freq, self.nb_iteration + self.report_iter_freq, self.report_iter_freq)
            self.mse["mse_batch"].fill(np.nan)

    def _save_parameter(self, name):
        hyper_parameter = {
            "learning_rate": self.learning_rate,
            "nb_input": self.nb_input,
            "nb_output": self.nb_output,
            "nb_time_step": self.nb_time_step,
            "nb_neuron": self.nb_neuron,
            "batch_size": self.batch_size,
            "nb_iteration": self.nb_iteration,
            "activation_fct": self.activation_fct
        }
        parameter = {
            "hyper_parameter": hyper_parameter,
            "mse": self.mse,
            "avg_mse_training": self.avg_mse_training,
            "seed": self.seed,
            "report_iter_freq": self.report_iter_freq,
        }
        with open(name, 'wb') as file:
            pickle.dump(parameter, file)

    def _load_parameter(self, name):
        with open(name, 'rb') as file:
            parameter = pickle.load(file)
        hyper_parameter = parameter["hyper_parameter"]
        self.learning_rate = hyper_parameter["learning_rate"]
        self.nb_input = hyper_parameter["nb_input"]
        self.nb_output = hyper_parameter["nb_output"]
        self.nb_time_step = hyper_parameter["nb_time_step"]
        self.nb_neuron = hyper_parameter["nb_neuron"]
        self.batch_size = hyper_parameter["batch_size"]
        self.nb_iteration = hyper_parameter["nb_iteration"]
        self.activation_fct = hyper_parameter["activation_fct"]
        self.mse = parameter["mse"]
        self.avg_mse_training = parameter["avg_mse_training"]
        self.seed = parameter["seed"]
        self.report_iter_freq = parameter["report_iter_freq"]

    def _create_graph(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_time_step, self.nb_input], name="X")
        y = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_time_step, self.nb_output], name="Y")
        hidden_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.nb_neuron, activation=self.activation_fct,
                                              initializer=tf.contrib.layers.variance_scaling_initializer(
                                                  factor=1.0, mode='FAN_AVG', uniform=False))
        output_cell = tf.contrib.rnn.OutputProjectionWrapper(hidden_cell, output_size=self.nb_output)
        output, state = tf.nn.dynamic_rnn(output_cell, inputs=x, dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(output - y), name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="opti")
        trainer = optimizer.minimize(loss, name="train")
        return trainer, loss

    def _eval_rnn(self, sess, data_handle, j):
        y_pred = self._predict(sess, data_handle.train_scaled[:self.nb_time_step],
                               data_handle.train_scaled[self.nb_time_step:],
                               data_handle.train_scaled[self.nb_time_step:].shape[0])
        self.mse["mse_training"][j] = np.mean(np.square(y_pred - data_handle.train_scaled[self.nb_time_step:]))
        y_pred = self._predict(sess, data_handle.test_scaled[:self.nb_time_step],
                               data_handle.test_scaled[self.nb_time_step:],
                               data_handle.test_scaled[self.nb_time_step:].shape[0])
        self.mse["mse_test"][j] = np.mean(np.square(y_pred - data_handle.test_scaled[self.nb_time_step:]))

    def train(self, data_handle, sess_file, seed=None, evaluate=False):
        """
        Train the RNN on the train set from the data_handle and
        save the trained model according to the given name
        :param data_handle: Object containing the train set and a next_batch method
        :param sess_file: Name of the model for the save
        :param seed: Seed used to initialize numpy and tensorflow Random state
        :param evaluate: If True, at each report_iter_freq, MSE will be evaluated on the completed test set and train set
        :return: List of the Mean Square Error every 'self.report_iteration_freq' step of the training
        """
        tf.reset_default_graph()
        self.seed = seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        trainer, loss = self._create_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            j = 0
            mse_tmp = 0
            for i in range(self.nb_iteration):
                x_batch, y_batch = data_handle.next_batch(self.batch_size, self.nb_time_step)
                sess.run(trainer, feed_dict={'X:0': x_batch, 'Y:0': y_batch})
                mse_tmp += loss.eval(feed_dict={'X:0': x_batch, 'Y:0': y_batch})
                if (i + 1) % self.report_iter_freq == 0:
                    self.mse["mse_batch"][j] = mse_tmp / self.report_iter_freq
                    mse_tmp = 0
                    delta = abs(self.mse["mse_batch"][j] - self.mse["mse_batch"][j - 5])
                    print("After {} iterations ; mse = {} ; delta = {}".format(i + 1, self.mse["mse_batch"][j], delta))
                    if delta < 0.0001:
                        print("CONVERGENCE !")
                    if np.isnan(self.mse["mse_batch"][j]):
                        self._nan = True
                        break
                    if evaluate:
                        self._eval_rnn(sess, data_handle, j)
                    j += 1
            y_pred = self._predict(sess, data_handle.train_scaled[:self.nb_time_step],
                                   data_handle.train_scaled[self.nb_time_step:],
                                   data_handle.train_scaled[self.nb_time_step:].shape[0])
            self.avg_mse_training = np.mean(np.square(y_pred - data_handle.train_scaled[self.nb_time_step:]))
            saver.save(sess, "./" + sess_file)
        self._save_parameter(sess_file + ".param")
        return self.mse

    def _predict(self, sess, input_set, test_set, nb_pred):
        x_batch = input_set.reshape(self.batch_size, self.nb_time_step, self.nb_input)
        y_pred = np.zeros(shape=(nb_pred, 1), dtype=np.float32)
        for i in range(nb_pred):
            x = x_batch[:, -self.nb_time_step:, :]
            result = sess.run("rnn/transpose:0", feed_dict={'X:0': x})
            y_pred[i] = result[0, -1, 0]
            if test_set is not None:
                x_batch = np.append(x_batch, test_set[i, 0].reshape(-1, 1, 1), axis=1)
            else:
                x_batch = np.append(x_batch, result[0, -1, 0].reshape(-1, 1, 1), axis=1)
        return y_pred

    def run(self, sess_file, input_set, test_set=None, nb_pred=1):
        """
        Run the RNN to predict value based on the input set.
        - If test_set is not is not given, nb_pred value(s) following of the input_set will be predicted
        - If test_set is given, test_set.shape[0] value(s) will be predicted.
        The input_set to make the next prediction is enriched after each prediction with the real value
        from the test_set.
        :param sess_file: Name of the file from which to load the meta graph
        :param input_set: Set of input data. The model will predict the value following the last of the input set
        :param test_set: Set of data containing the real value of what the prediction will try to guess.
        If provided, the test_set will be used to complete the input_set for the next prediction.
        :param nb_pred: Number of prediction value to be made after the input_set
        :return:
        """
        if input_set.shape[0] != self.nb_time_step:
            raise ValueError("The input_set shall have exactly self.nb_time_step (={}) rows. "
                             "Got {} rows".format(self.nb_time_step, input_set.shape[0]))
        if test_set is not None and nb_pred != 1:
            raise AttributeError("Can't define both, test_set and nb_pred.")
        nb_pred = test_set.shape[0] if test_set is not None else nb_pred
        if nb_pred <= 0:
            raise ValueError("nb_pred shall greater or equal to 1")
        if self._nan:
            return np.array([np.nan] * nb_pred).reshape(-1, 1)
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(sess_file + ".meta")
            saver.restore(sess, sess_file)
            y_pred = self._predict(sess, input_set, test_set, nb_pred)
        return y_pred
