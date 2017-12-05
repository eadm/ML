import csv
import numpy as np


def read_data(path):
    x, y = [], []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            x.append(row['x'])
            y.append(row['y'])
    return np.array(x), np.array(y)
