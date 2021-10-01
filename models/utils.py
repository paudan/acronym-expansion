import csv
from collections import OrderedDict, Iterable
from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import CSVLogger


def handle_value(k):
    is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
    if isinstance(k, Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
    else:
        return k

class KerasCSVLogger(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not self.writer:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch', 'time'] + self.keys, dialect=csv.excel)
            if self.append_header:
                self.writer.writeheader()
        row_dict = OrderedDict({'epoch': epoch, 'time': str(datetime.now())})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()