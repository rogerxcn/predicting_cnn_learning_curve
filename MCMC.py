import pymc3 as pm
import curve_def
import pandas as pd
from scipy.optimize import curve_fit
import theano.tensor as tt

def MCMC_curve_fit(func, input_data, label, default_para, N_SAMPLES):
    """
    :param default_para: list of default value
    """
    with pm.Model() as _:
        parameter = []
        for para_idx in range(len(default_para)):
            parameter.append(
                pm.Normal(
                    'parameter_'+str(para_idx),
                    mu=default_para[para_idx],
                    tau=0.64#
                ),
            )
        observed = pm.Normal(
            'obs',
            mu=pm.Deterministic(
                'mean',
                func(input_data, *parameter)
            ),
            #tau=pm.Normal(
            #    'tau',
            #    mu=default_tau,#
            #    tau=0.05,#
            #    testval=0.0
            #),
            tau=0.64,#
            observed=label
        )
        trace = pm.sample(N_SAMPLES, step=pm.Metropolis(), progressbar=False)

        MCMC_result = []
        for para_idx in range(len(default_para)):
            MCMC_result.append((trace['parameter_'+str(para_idx)][N_SAMPLES:, None]).mean())
    return trace, MCMC_result

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

ABS_USED = False
ABS_EVAL_USED = True

def main():
    data = pd.read_csv("./data/data.csv")
    epoch = data["epoch"].values
    names = list(data)[1:]
    x = epoch[:(-1-AVG_SMOOTHING)]

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

    train_data = data[train[0]].values[:-2]

    popt,  pcov = curve_fit(curve_def.ILog2, x, train_data)
    _, MCMC_result = MCMC_curve_fit(curve_def.ILog2, x, train_data, popt, 10000)
    # Pow3 Janoschek
    # Exp4
    print("MCMC", MCMC_result)
    print("curve_fit", popt)
    return

if __name__=='__main__':
    main()


