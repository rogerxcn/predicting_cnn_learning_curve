import curve_def
import collections
import random
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

AVG_SMOOTHING = 1
PRED_LEN = 100
TEST_POINTS = 3
COMB_NUM_POINTS = 6
NUM_CLUSTER = 6

INPUT_START_OFFSET = 0

VALID_NORMALIZATION_MODE = ["mean", "offset"]
NORMALIZATION_MODE = "mean"
OFFSET_INDEX = 0

VALID_REFIT_MODE = ["full", "pred", "scale"]
REFIT_MODE = "full"
DATA_DIR = "D:\\pycharm\\WORKS\\CS229\\cs229_curve_prediction\\data\\"

ABS_USED = False
ABS_EVAL_USED = True


###############################################
## Main
###############################################

def main():
    assert NORMALIZATION_MODE in VALID_NORMALIZATION_MODE, "Normalization mode not valid"
    assert REFIT_MODE in VALID_REFIT_MODE, "Re-fit mode not valid"

    infile = "data.csv"
    data = pd.read_csv(DATA_DIR+infile)

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

    # split data
    train = []
    dev = []
    test = []

    for i in range(len(names)):
        if i % 5 < 2:
            train.append(names[i])
        elif i % 5 < 4:
            dev.append(names[i])
        else:
            test.append(names[i])

    print("Train split:", train)
    print("Dev split:", dev)
    print("Test split:", test)

    # begin main algorithm

    cluster_data = []
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
        cluster_data.append(train_input_normalized.tolist())
    cluster_data = np.array(cluster_data)
    cluster_centers = KMeans(n_clusters=NUM_CLUSTER).fit_predict(cluster_data)

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

        cluster_data = []
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

            dev_loss.append(np.sum(np.abs(train_input_normalized - dev_input_normalized)))

        best_match = np.argmin(np.array(dev_loss))
        best_center = cluster_centers[best_match]

        yhat = None
        if REFIT_MODE == "full":
            fit_loss = []

            # prepare the data
            fit_inputx = []
            fit_inputy = []
            for idx in range(len(cluster_centers)):
                if cluster_centers[idx] == best_center:
                    fit_inputx += x.tolist()
                    fit_inputy += (data[train[idx]].values[:-2]).tolist()

            for fit_func in fit_funcs:
                y = data[train[best_match]].values[:-2]
                popt, _ = curve_fit(fit_func, fit_inputx, fit_inputy)
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
                y = data[train[best_match]].values[:-2]
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
            y = data[train[best_match]].values[:-2]

            if NORMALIZATION_MODE == "mean":
                train_input_mean = np.mean(train_input)
                train_input_diff = np.max(train_input) - np.min(train_input)
                train_input_normalized = (train_input - train_input_mean) / train_input_diff

            elif NORMALIZATION_MODE == "offset":
                train_input_mean = np.mean(train_input)
                train_input_diff = train_input[OFFSET_INDEX]
                train_input_normalized = (train_input - train_input_mean) / train_input_diff

            yhat = train_input_normalized * dev_input_diff + dev_input_mean

        if ABS_EVAL_USED:
            current_loss = np.sum(np.abs(yhat[-TEST_POINTS] - dev_test))
        else:
            current_loss = np.sum(np.abs(np.mean(yhat[-TEST_POINTS]) - np.mean(dev_test)))

        loss.append(current_loss)

    loss = np.array(loss)
    print("Max loss:", 100* np.max(loss), "%")
    print("Min loss:", 100 * np.min(loss), "%")
    print("Average loss:", 100*np.mean(loss), "%")
    print("Fit function chosen:", fit_func_chosen)
    print("loss list: ", loss)


if __name__=='__main__':
    main()

