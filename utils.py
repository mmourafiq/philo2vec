# -*- coding: utf-8 -*-

import os
import json
import time


def time_(function):

    def decorator(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()

        print('{} {:,.2f} sec'.format(function.__name__, end - start))
        return result

    return decorator


def get_data():
    data_dir = 'philo2vec/data/data/'
    result = []
    for filename in os.listdir(data_dir):
        if filename.startswith('.'):
            continue

        with open(data_dir + filename, 'r') as f:
            content = json.loads(f.read())
            content = ' '.join(' '.join([i for i in content if i != 'li']).split('\n'))
            content = content.encode('ascii', 'ignore')
            result += content.split('.')

    return result
