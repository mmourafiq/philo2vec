import pandas as pd
import time


def time_(function):

    def decorator(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()

        print('{} ({}, {}) {:,.2f} sec'.format(function.__name__, args, kwargs, end - start))
        return result

    return decorator


def get_data():
    import os
    import json
    data_dir = 'philo2vec/data/data/'
    result = []
    for filename in os.listdir(data_dir):
        if filename.startswith('.'):
            continue
        with open(data_dir + filename, 'r') as f:
            content = f.read()
            result += ' '.join(' '.join([i for i in json.loads(content) if i != 'li']).split('\n')).split('.')

    return pd.Series(result)
