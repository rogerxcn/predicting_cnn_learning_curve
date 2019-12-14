import collections
import random
import json

import curve_def

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

AVG_SMOOTHING = 1
PRED_LEN = 100
TEST_POINTS = 3
COMB_NUM_POINTS = 6


###############################################
## Main
###############################################

def main():
    infile = "data.csv"
    data = pd.read_csv(infile)

    # read epoch data
    epoch = data["epoch"].values
    # read all accuracy data
    names = list(data)[1:]
    data_size = len(names)
    print("Data length: ", data_size)
    # get all possible functions
    fit_funcs = curve_def.curve_set()
    # extract x coodinates data
    x = epoch[:(-1 - AVG_SMOOTHING)]

    train = ['2L-1', '2L-3', '2L-4', '2L-6', '2L-8', '2L-10', '2L-12', '2L-14',  '3L-8', '3L-10', '3L-11', '2L-7', '2L-11', '2L-13', '3L-7', '3L-3', '3L-5']
    test = ['2L-5', '2L-9', '2L-15', '3L-9', '3L-4', '2L-17', '2L-16', '2L-18', '2L-19', '3L-1', '3L-2',  '3L-6',]

    # coloring for plots
    colors = iter(plt.cm.rainbow(np.linspace(0,1,32)))

    # dictionary: fit function -> total fitting error
    err = collections.defaultdict(float)

    # for each curve
    for name in train:

        name = "2L-3"

        # dictionary: fit function -> fitting error for the current curve
        func_to_err = {}

        best_input_loss = 0xFFFF
        best_err = 0

        # for each fit function
        for fit_func in fit_funcs:

            fit_func = fit_funcs[4]

            # raw data
            raw_points = data[name].values[:-2]
            # sliding window average of size AVG_SMOOTHING
            smoothed_points = np.convolve(raw_points, np.ones((AVG_SMOOTHING,))/AVG_SMOOTHING, mode='valid')

            # fit the first PRED_LEN data points
            popt, _ = curve_fit(fit_func, x[:PRED_LEN], smoothed_points[:PRED_LEN])

            # evaluate the fitness of the fit function
            xprime = np.array([(150 - TEST_POINTS + a) for a in range(TEST_POINTS)])
            estimate = fit_func(xprime, *popt)
            true_val = raw_points[-TEST_POINTS:]

            loss = np.sum(np.abs(estimate-true_val))

            # record the loss
            func_to_err[fit_func.__name__] = loss

            # if we were to use the best fitting function
            input_x = x[(PRED_LEN - COMB_NUM_POINTS):PRED_LEN]
            input_y = smoothed_points[(PRED_LEN - COMB_NUM_POINTS):PRED_LEN]
            pred_y = fit_func(input_x, *popt)

            plt.axvline(x=100, linestyle="--", color="grey")
            plt.plot(x[:99], fit_func(x, *popt)[:99], label="fit area",color="green", alpha=0.7)
            plt.plot(x[99:], fit_func(x, *popt)[99:], linestyle="--", label="prediction",color="green", alpha=0.7)
            plt.scatter(x, raw_points, label="target", marker=".", color="orange", alpha=0.7)

            plt.xticks(np.arange(0, 150.1, 10))
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Direct Curve Fitting Prediction")
            plt.legend()
            plt.savefig("curve_fit.png")
            plt.show()

            return

            # calculate the loss
            input_loss = np.sum(np.abs(pred_y-input_y))

            # record the best loss and the corresponding error
            if input_loss < best_input_loss:
                best_input_loss = input_loss
                best_err = loss

        # add combined method loss
        func_to_err["comb"] = best_err

        print(best_err)

        # add each loss to their corresponding fit function
        for func in func_to_err:
            err[func] += func_to_err[func]

    # evaluate the loss for each fit function (use json for pretty print)
    sorted_err = sorted(err.items(), key=lambda a: a[1])

    print("Total error on each fit function: ")
    print(json.dumps(err, indent=4))

    print("Average error on each fit function: ")
    avg_err = {k : v / data_size for k, v in sorted_err}
    print(json.dumps(avg_err, indent=4))



if __name__=='__main__':
    main()
