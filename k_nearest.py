import curve_def

import collections
import random

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


AVG_SMOOTHING = 1
PRED_LEN = 100
TEST_POINTS = 3
COMB_NUM_POINTS = 6

INPUT_START_OFFSET = 0

VALID_NORMALIZATION_MODE = ["mean", "offset"]
NORMALIZATION_MODE = "mean"
OFFSET_INDEX = 0

VALID_REFIT_MODE = ["full", "pred", "scale"]
REFIT_MODE = "full"

ABS_USED = False
ABS_EVAL_USED = True

K = 3


###############################################
## Main
###############################################

def main():
    assert NORMALIZATION_MODE in VALID_NORMALIZATION_MODE, "Normalization mode not valid"
    assert REFIT_MODE in VALID_REFIT_MODE, "Re-fit mode not valid"

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

    # coloring for plots
    colors = iter(plt.cm.rainbow(np.linspace(0,1,32)))

    print(names)

    # split data
    train = ['2L-1', '2L-3', '2L-4', '2L-6', '2L-8', '2L-10', '2L-12', '2L-14',  '2L-16', '2L-18', '2L-19', '3L-1', '3L-2',  '3L-6', '3L-8', '3L-10', '3L-11']
    dev = ['2L-7', '2L-11', '2L-13', '3L-7', '3L-3', '3L-5']
    test = ['2L-5', '2L-9', '2L-15', '3L-9', '3L-4', '2L-17']

    # begin main algorithm

    loss = []
    fit_func_chosen = []

    # for each dev data
    for dev_name in dev:
        dev_loss = []
        dev_data = data[dev_name].values[:-2]

        dev_test = dev_data[-TEST_POINTS:]
        dev_input = dev_data[INPUT_START_OFFSET:PRED_LEN]

        if NORMALIZATION_MODE == "mean":
            dev_input_mean = np.mean(dev_input)
            dev_input_diff = np.max(dev_input) - np.min(dev_input)
            dev_input_normalized = (dev_input - dev_input_mean) / dev_input_diff
        elif NORMALIZATION_MODE == "offset":
            dev_input_mean = np.mean(dev_input)
            dev_input_diff = dev_input[OFFSET_INDEX]
            dev_input_normalized = (dev_input - dev_input_mean) / dev_input_diff



        for train_name in train:
            train_data = data[train_name].values[:-2]

            train_input = train_data[INPUT_START_OFFSET:PRED_LEN]

            if NORMALIZATION_MODE == "mean":
                train_input_mean = np.mean(train_input)
                train_input_diff = np.max(train_input) - np.min(train_input)
                train_input_normalized = (train_input - train_input_mean) / train_input_diff
            elif NORMALIZATION_MODE == "offset":
                train_input_mean = np.mean(train_input)
                train_input_diff = train_input[OFFSET_INDEX]
                train_input_normalized = (train_input - train_input_mean) / train_input_diff


            dev_loss.append((np.sum(np.abs(train_input_normalized - dev_input_normalized)), train_name))

        sorted_dev_loss = sorted(dev_loss, key=lambda a: a[0])
        best_matches = [a[1] for a in sorted_dev_loss[:K]]
        weight = [a[0] for a in sorted_dev_loss[:K]]

        weight = np.exp(-np.array(weight))
        weight /= np.sum(weight)

        weight_dict = { best_matches[i] : weight[i] for i in range(len(best_matches))}

        weighted_yhat = np.zeros(len(dev_data))

        for best_match in best_matches:
            yhat = None

            if REFIT_MODE == "full":
                fit_loss = []

                for fit_func in fit_funcs:
                    y = data[best_match].values[:-2]
                    popt, _ = curve_fit(fit_func, x, y)
                    yhat = fit_func(x, *popt)

                    if ABS_USED:
                        fit_loss.append(np.sum(np.abs(y[-TEST_POINTS:] - yhat[-TEST_POINTS:])))
                    else:
                        fit_loss.append(np.sum(np.abs(np.mean(y[-TEST_POINTS:]) - np.mean(yhat[-TEST_POINTS:]))))

                best_idx = np.argmin(np.array(fit_loss))
                best_fit_func = fit_funcs[best_idx]

                popt, _ = curve_fit(best_fit_func, x[:PRED_LEN], dev_data[:PRED_LEN])
                yhat = best_fit_func(x, *popt)

                fit_func_chosen.append(best_fit_func.__name__)

            elif REFIT_MODE == "pred":
                fit_loss = []

                for fit_func in fit_funcs:
                    y = data[best_match].values[:-2]
                    popt, _ = curve_fit(fit_func, x[:PRED_LEN], y[:PRED_LEN])
                    yhat = fit_func(x, *popt)

                    if ABS_USED:
                        fit_loss.append(np.sum(np.abs(y[-TEST_POINTS:] - yhat[-TEST_POINTS:])))
                    else:
                        fit_loss.append(np.sum(np.abs(np.mean(y[-TEST_POINTS:]) - np.mean(yhat[-TEST_POINTS:]))))

                best_idx = np.argmin(np.array(fit_loss))
                best_fit_func = fit_funcs[best_idx]

                popt, _ = curve_fit(best_fit_func, x[:PRED_LEN], dev_data[:PRED_LEN])
                yhat = best_fit_func(x, *popt)

                fit_func_chosen.append(best_fit_func.__name__)

            elif REFIT_MODE == "scale":
                y = data[best_match].values[:-2]

                if NORMALIZATION_MODE == "mean":
                    train_input_mean = np.mean(train_input)
                    train_input_diff = np.max(train_input) - np.min(train_input)
                    train_input_normalized = (train_input - train_input_mean) / train_input_diff

                elif NORMALIZATION_MODE == "offset":
                    train_input_mean = np.mean(train_input)
                    train_input_diff = train_input[OFFSET_INDEX]
                    train_input_normalized = (train_input - train_input_mean) / train_input_diff

                yhat = train_input_normalized * dev_input_diff + dev_input_mean

            weighted_yhat += weight_dict[best_match] * np.array(yhat)

        if ABS_EVAL_USED:
            current_loss = np.sum(np.abs(yhat[-TEST_POINTS] - dev_test))
        else:
            current_loss = np.sum(np.abs(np.mean(yhat[-TEST_POINTS]) - np.mean(dev_test)))

        loss.append(current_loss)

    loss = np.array(loss)
    print(loss)
    print("Max loss:", np.max(loss)/3)
    print("Min loss:", np.min(loss)/3)
    print("Average loss:", np.mean(loss)/3)
    print("Fit function chosen:", fit_func_chosen)


if __name__=='__main__':
    main()
