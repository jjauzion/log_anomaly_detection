from elasticsearch import Elasticsearch as es
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt


class Log:
    """
    log object contains aggregated data of log count per time bucket.

    Methods:
        - import_data() -> create a log object by sending a query to Kibana
        - load() -> load a a log previously saved
        - save() -> save the current log
    """

    def __init__(self):
        self.data = []

    def import_data(self,
                    backend_tag_keyword="",
                    lte="",
                    gte="",
                    query="default",
                    address="https://datalab.42cloud.io:9200",
                    user="jjauzion",
                    pwd=os.environ.get("private")):
        """
        Import data from Kibana, see query in the code.
        :param backend_tag_keyword: Filter log based on the backend tag keyword
        :param lte: Request log prior to lte date
        :param gte: Request log older than gte date
        :param query: Query to be sent to Kibana. See default query in the code
        :param address: address of the Kibana server
        :param user: user name
        :param pwd: user password
        :return: No return. Loaded data are stored in a structured
        numpy array in self.data with column "time" and "count"
        """
        if query != "default" and (backend_tag_keyword != "" or lte != "" or gte != ""):
            raise AttributeError("Cannot specify lte, gte, backend_tag_keyword if query is specified")
        if query == "default" and not isinstance(lte, datetime.datetime):
            raise TypeError("lte arg shall be datetime.datetime object (got {})".format(type(lte)))
        if query == "default" and not isinstance(gte, datetime.datetime):
            raise TypeError("gte arg shall be datetime.datetime object (got {})".format(type(gte)))
        if backend_tag_keyword == "":
            raise ValueError("backend_tag_keyword is empty")
        lte = lte.strftime("%d/%m/%Y")
        gte = gte.strftime("%d/%m/%Y")
        index = "varnishlogbeat"
        aggs_name = "byhour"
        if query == "default":
            query = {
                "size": 0,
                "query": {
                   "bool": {
                     "must": [
                       {
                         "match": {
                          "backend.tag.keyword": {
                            "query": backend_tag_keyword
                            }
                         }
                        },
                        {
                          "range": {
                            "@timestamp": {
                              "gte": gte,
                              "lte": lte,
                              "format": "dd/MM/yyyy"
                            }
                          }
                        }
                      ]
                   }
                 },
                "aggs": {
                    aggs_name: {
                        "date_histogram": {
                            "field": "@timestamp",
                            "interval": "hour"
                        }
                    }
                }
            }
        elastic = es(address, http_auth=[user, pwd], verify_certs=False,
                     use_ssl=True)
        result = elastic.search(index=index, body=query)
        result = result["aggregations"][aggs_name]["buckets"]
        self.data = np.zeros(len(result), dtype=[('count', np.float64), ('time', 'datetime64[ms]')])
        for i, element in enumerate(result):
            date = element["key_as_string"]
#            print("{} ; {}".format(element["doc_count"], element["key_as_string"]))
            if date[-1] == 'Z':
                date = date[:-1] + "+0000"
            date = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f%z")
            date = date.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
            self.data[i]['time'] = date
            self.data[i]['count'] = element["doc_count"]

    def save(self, name, directory=""):
        name = directory + name
        np.save(name, self.data)

    def load(self, src):
        self.data = np.load(src)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.data['time'], self.data['count'])

    def print_sample(self, i_start, i_end):
        for index in range(i_end - i_start):
            print("i:{} ; count:{} ; date:{}".format(index, self.data['count'][index], self.data['time'][index]))

    def concat(self, log_1, log_2):
        """
        Concatenate data from two log and save it in self.data
        :param log_1:
        :param log_2:
        :return:
        """
        if not hasattr(log_1, 'data') or not hasattr(log_2, 'data'):
            raise AttributeError("log_1 or log_2 has no 'data' attribute")
        if "time" not in log_1.data.dtype.names or "count" not in log_1.data.dtype.names:
            raise AttributeError("log_1 has no 'time' or 'count' column")
        if "time" not in log_2.data.dtype.names or "count" not in log_2.data.dtype.names:
            raise AttributeError("log_2 has no 'time' or 'count' column")
        self.data = np.concatenate((log_1.data, log_2.data))
