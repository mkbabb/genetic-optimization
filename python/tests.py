import numpy as np


def flatten_list(seq: list,
                 order: int = 1) -> list:
    global shape, ndim
    shape = [len(seq)]
    ndim = 0

    def recurse(seq: list) -> list:
        global shape, ndim
        ndim = len(seq)
        buff = []
        ndim = 0
        for i in range(ndim):
            seq_i = seq[i]
            if isinstance(seq_i, list):
                buff_r = recurse(seq_i)
                M = len(seq_i)
                if len(shape) <= ndim + 1:
                    shape.insert(-1, M)
                ndim += 1
                buff += [buff_r] if ndim <= order else buff_r
            elif ndim != 0:
                buff += [seq_i]
        if ndim == 0:
            return seq
        else:
            return buff

    flat = recurse(seq)
    return flat, ndim, shape


def get_ndim(seq):
    global ndim
    ndim = 0

    def recurse(seq):
        global ndim
        ndim = len(seq)
        ndim = 0
        for i in range(ndim):
            seq_i = seq[i]
            if (isinstance(seq_i, list)):
                recurse(seq_i)
                ndim += 1
    recurse(seq)
    return ndim + 1


def pprint(seq, formatter, sep=", "):
    global prev_ix
    prev_ix = 0
    prev_len = 0
    ndim = get_ndim(seq)

    def recurse(seq, ix):
        global prev_ix, prev_len
        prev_ix = ix
        buff_s = ""
        ndim = len(seq)
        for i in range(ndim):
            seq_i = seq[i]
            if (isinstance(seq_i, list)):
                buff_s += recurse(seq_i, ix + 1)
            else:
                buff_s += formatter(seq_i)
                buff_s += sep if i < ndim - 1 else ""
        if (prev_ix > ix):
            buff_s = buff_s[:-(prev_len)]

        buff_s = f"{{{buff_s}}}"

        if (ix > 0):
            new_line = "\n" * (ndim - ix)
            hanging_indent = " " * (ix)
            spacing = sep.strip() + new_line + hanging_indent
            prev_len = len(spacing)
            buff_s += spacing

        return buff_s

    s = recurse(seq, 0)
    print(s)
    return s


# l = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12], [5, 6]]]

N = 2*3*80
t = np.arange(N).reshape((2, 3, 80))

# l = [[[1, 2, 3], [4, 5, 6]],
#      [[7, 8, 9], [10, 11, 12]]]


pprint(t.tolist(), lambda x: f"{x}")
