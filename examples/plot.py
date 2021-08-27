#!/usr/bin/env python
"""
Script for plotting results.

```
python plot.py logs.tsv
```
"""
import argparse
import glob
import os

import gnuplotlib as gp
import numpy as np
import pandas  # Fast CSV reading.


parser = argparse.ArgumentParser()

parser.add_argument("--xkey", default="step", type=str, help="x values to plot.")
parser.add_argument(
    "--ykey", default="episode_return", type=str, help="y values to plot."
)
parser.add_argument("--window", default=50, type=int, help="Smoothing window size.")
parser.add_argument("--width", default=80, type=int, help="Width of plot.")
parser.add_argument("--height", default=30, type=int, help="Height of plot.")
parser.add_argument(
    "--errorbars", action="store_true", help="Whether to print error bars."
)
parser.add_argument(
    "--smoothing",
    default="pandas",
    choices=["pandas", "convolve", "cumsum"],
    help="Smoothing algorithm.",
)
parser.add_argument("files", nargs="*", type=str)


def moving_average_cumsum(a, n=20):
    # Fast, but doesn't play well with NaNs
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def moving_average(a, n=20):
    return np.convolve(a, np.ones((n,)) / n, mode="valid")


def rolling_xs_ys(xs, ys, window_size=20):
    """Alternative to rolling() in pandas."""
    ma = moving_average_cumsum if FLAGS.smoothing == "cumsum" else moving_average
    return xs[window_size - 1 :], ma(ys, window_size)


def plot(xys, xrange=None, yrange=None, color="green"):
    plot_options = dict(
        terminal="dumb %d %d ansi" % (FLAGS.width, FLAGS.height),
        title=FLAGS.ykey,
        xlabel=FLAGS.xkey,
        set=("key outside bottom center",),
        # _with="points linecolor '%s'" % color,
    )

    if FLAGS.errorbars:
        plot_options["with"] = "yerrorbars"

    if xrange is not None:
        plot_options.update(xrange=xrange)

    if yrange is not None:
        plot_options.update(yrange=yrange)

    gp.plot(*xys, **plot_options)


def load_file(filename):
    delimiters = {".tsv": "\t", ".csv": ","}
    _, ext = os.path.splitext(filename)

    if ext not in delimiters:
        raise RuntimeError("Filetype not recognised (expected .csv or .tsv): %s" % ext)

    df = pandas.read_csv(filename, sep=delimiters[ext])

    xs = np.array(df[FLAGS.xkey])

    if FLAGS.smoothing == "pandas":
        window = df[FLAGS.ykey].rolling(window=FLAGS.window, min_periods=0)
        ys = np.array(window.mean())
    else:
        ys = np.array(df[FLAGS.ykey])
        xs, ys = rolling_xs_ys(xs, ys, window_size=FLAGS.window)

    return (xs, ys, {"legend": filename})


def main():
    xys = []

    for pattern in FLAGS.files:
        for filename in glob.glob(pattern):
            xys.append(load_file(filename))

    plot(xys)


if __name__ == "__main__":
    global FLAGS
    FLAGS = parser.parse_intermixed_args()
    main()
