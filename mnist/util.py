import torch
import numpy as np


def test():
    pass


def dict_string(my_dict):
    str_list = []
    for key in my_dict:
        if type(my_dict[key]) == float:
            str_list.append(key + f"_{my_dict[key]:.6f}")
        else:
            str_list.append(key + f"_{my_dict[key]}")

    return "_".join(str_list)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

    # use: list(product_dict(**mydict))


def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(gen)


if __name__ == '__main__':
    start_time = time.time()
    test()
    duration = (time.time() - start_time)
    print("---train_cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))