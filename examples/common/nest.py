# Copyright (c) Facebook, Inc. and its affiliates.


def map(f, n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n.__class__(map(f, sn) for sn in n)
    elif isinstance(n, dict):
        return {k: map(f, v) for k, v in n.items()}
    else:
        return f(n)


def flatten(n):
    if isinstance(n, tuple) or isinstance(n, list):
        for sn in n:
            yield from flatten(sn)
    elif isinstance(n, dict):
        for key in n:
            yield from flatten(n[key])
    else:
        yield n


def zip(*nests):
    n0, *nests = nests
    iters = [flatten(n) for n in nests]

    def f(first):
        return [first] + [next(i) for i in iters]

    return map(f, n0)


def map_many(f, *nests):
    n0, *nests = nests
    iters = [flatten(n) for n in nests]

    def g(first):
        return f([first] + [next(i) for i in iters])

    return map(g, n0)
