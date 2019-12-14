import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# epoch: np.array(dataFrame['epoch'] = 1,2,3,...,150)
# data: np.array(list(dataFrame['2L-1']))

def Vapor_Pressure(x, a, b, c):
    return np.exp(a + b/x + c*np.log(x))
def Pow3(x, c, a, alpha):
    return c - a * np.power(x, -alpha)
def Log_Log_Linear(x, a, b):
    return np.log(np.abs(a * np.log(x) + b))
def Log_Power(x, a, b, c):
    return a/(1+np.power(x/np.exp(b), c))
def Pow4(x, a, b, c, alpha):
    return c - np.power(np.abs(a*x+b), -alpha)
def MMF(x, alpha, beta, delta, kar):
    return alpha - (alpha-beta)/(1+np.power(np.abs(kar*x), delta))
def Exp4(x, a, b, c, alpha):
    return c - np.exp(-a*np.power(x, alpha) + b)
def Janoschek(x, alpha, beta, delta, kar):
    return alpha - (alpha - beta)*np.exp(-kar*np.power(x, delta))
def Weibull(x, alpha, beta, delta, kar):
    return alpha - (alpha - beta) * np.exp(-np.power(np.abs(kar*x), delta))
def ILog2(x, a, b):
    return b - a/(np.log(x))


def main(curve_file, Func, plot_flag=False):
    dataFrame = pd.read_csv(curve_file)
    pcov_list = []
    loss_list = []
    for head in list(dataFrame):
        if head == 'epoch':
            continue
        popt, pcov = curve_fit(Func, np.array(dataFrame['epoch']), np.array(dataFrame[head]))
        #popt, pcov = curve_fit(Func, np.array(dataFrame['epoch'])[1:], np.array(dataFrame[head])[1:]) # for ILog2 only
        y_ = Func(np.array(dataFrame['epoch']), *popt)
        #y_ = Func(np.array(dataFrame['epoch'])[1:], *popt) # for ILog2 only
        pcov_list.append(pcov)
        loss_list.append((y_ - np.array(dataFrame[head])) ** 2)
        # loss_list.append((y_ - np.array(dataFrame[head])[1:]) ** 2) # for ILog2 only

        with open('./' + str(Func).split()[1] + '.cov', 'w') as cov_file:
            cov_file.write(str(pcov_list))
        with open('./' + str(Func).split()[1] + '.loss', 'w') as loss_file:
            loss_file.write(str(loss_list))

        plt.plot(np.array(dataFrame['epoch']), np.array(dataFrame[head]), 's', markersize=2, label=head)
        plt.plot(np.array(dataFrame['epoch']), y_, linewidth=2)
        # plt.plot(np.array(dataFrame['epoch'])[1:], y_, linewidth=2) # for ILog2 only

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc=4, fontsize=3.5)
    plt.title('curve fit --- ' + str(Func).split()[1])
    plt.savefig('./' + str(Func).split()[1] + '.png')
    if plot_flag:
        plt.show()
    return None

if __name__=='__main__':
    curve_file = "D:\\pycharm\\WORKS\\CS229\\FinalPJ\\data.csv"
    Func_list = [Vapor_Pressure, Pow3, Log_Log_Linear, Log_Power, Pow4, MMF, Exp4, Janoschek, Weibull, ILog2]
    # Max_idx = 9
    # Weibull(error), ILog2(error)
    main(curve_file, Janoschek, True)


    # plot 2D Gaussian
"""
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import math

num = 500
X, Y = np.meshgrid(np.linspace(-0.05, 0.05, num), np.linspace(-0.05, 0.05, num))

u = np.array([0, 0])
o = np.array([[1.55976158e-04, 4.22484277e-05],
       [4.22484277e-05, 1.43301084e-05]])  # Log_Log_Linear_3L-11

pos = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)

a = np.dot((pos - u), np.linalg.inv(o))
b = np.expand_dims(pos - u, axis=3)
# Z = np.dot(a.reshape(200*200,2),(pos-u).reshape(200*200,2).T)
Z = np.zeros((num, num), dtype=np.float32)
for i in range(num):
    Z[i] = [np.dot(a[i, j], b[i, j]) for j in range(num)]

Z = np.exp(Z * (-0.5)) / (2 * np.pi * math.sqrt(np.linalg.det(o)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

cset = ax.contour(X, Y, Z, 10, zdir='z', offset=0, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-4, cmap=mpl.cm.winter)
cset = ax.contour(X, Y, Z, zdir='y', offset=4, cmap=mpl.cm.winter)

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('')

plt.title('ILog2 V19-4 Parameter Distribution')
plt.show()
"""

"""
Log_Log_Linear_V19_4 = array([9.97966431e-04, 1.96482331e-03, 2.07666462e-03, 1.83506471e-03,
       1.49162840e-03, 9.73712374e-04, 5.55826375e-04, 3.91146336e-04,
       1.66850240e-04, 1.50570546e-04, 1.52114290e-06, 8.13577410e-06,
       1.58301389e-05, 1.01258594e-04, 1.07941513e-06, 3.74612164e-04,
       4.73841694e-04, 2.57370846e-04, 5.08511295e-04, 4.06644757e-04,
       4.06727889e-04, 2.41441143e-04, 2.74317518e-04, 4.59093056e-04,
       3.03398249e-04, 4.94073473e-04, 5.03803536e-04, 4.43674598e-04,
       4.65460429e-04, 5.84280162e-04, 8.38090161e-04, 4.88539768e-04,
       4.29589513e-04, 5.25154513e-04, 3.94741235e-04, 1.58217563e-04,
       5.97509779e-04, 6.03278105e-04, 1.80331987e-04, 5.92561725e-04,
       3.34918535e-04, 1.96031607e-04, 5.40171304e-04, 3.28343194e-04,
       1.30766629e-04, 3.64243820e-04, 3.30085052e-04, 2.81672025e-04,
       2.34958348e-04, 1.41675762e-04, 2.99467473e-04, 1.35357534e-04,
       3.41854112e-04, 1.70799983e-04, 2.42506929e-04, 2.59183297e-04,
       2.05859849e-04, 2.79484080e-04, 1.79782476e-04, 8.49838130e-05,
       1.91772709e-04, 9.99252499e-05, 1.92161477e-04, 3.53493259e-05,
       1.45096479e-04, 5.27353789e-05, 1.87524640e-04, 9.10358380e-05,
       6.89459914e-05, 7.88513753e-05, 9.54572289e-05, 6.35869881e-05,
       1.29759761e-04, 2.01941017e-06, 7.85585563e-05, 4.37936227e-05,
       8.05303608e-08, 1.56920885e-05, 1.98025880e-05, 2.25581922e-05,
       2.07904574e-05, 1.14248628e-05, 8.00531486e-09, 1.26080679e-05,
       1.60479165e-07, 6.02135606e-05, 3.28098119e-05, 3.72724811e-05,
       1.37571630e-05, 1.72032610e-07, 4.73091032e-07, 6.24250575e-06,
       2.51713389e-05, 7.05724235e-06, 4.50553146e-06, 1.01576579e-07,
       1.34907710e-07, 1.58166584e-05, 1.16579938e-05, 8.09367764e-06,
       3.80544846e-05, 1.47332414e-06, 1.51878445e-05, 2.48151542e-07,
       5.04156578e-05, 9.42990603e-08, 2.58016142e-05, 7.40190672e-07,
       3.76440172e-05, 1.85332169e-05, 6.35065284e-05, 1.56945813e-04,
       3.19738413e-04, 1.10869985e-04, 9.35601167e-05, 2.51083191e-05,
       1.07000140e-04, 2.68059906e-04, 1.43909964e-04, 1.56632920e-04,
       1.94039144e-04, 9.29254270e-05, 1.30996121e-04, 1.57416613e-04,
       4.72781000e-04, 9.67522879e-05, 6.06384978e-04, 3.20747628e-04,
       3.23638980e-04, 1.29198921e-04, 1.99919669e-04, 9.73952021e-04,
       2.09478071e-04, 5.68098264e-04, 2.36932915e-04, 2.83816449e-04,
       1.98740875e-04, 5.08266458e-04, 2.55633922e-04, 6.98490467e-04,
       2.77757490e-04, 1.20408083e-03, 6.56919829e-04, 4.82146859e-04,
       3.34235171e-04, 4.08171827e-04, 2.79604398e-04, 4.25864328e-04,
       1.34312133e-03, 5.22480574e-04])
plt.plot(Log_Log_Linear_V19_4)
plt.xlabel('epoch')
plt.ylabel('fitting loss')
plt.title('Log-Log-Linear 3L-11 fitting loss')
"""
