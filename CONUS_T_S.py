import numpy as np
import pandas as pd
import os
import sys
sys.path.append('G:\Farshid\SNTemp_newData\example')
import matplotlib.pyplot as plt
import hydroDL


from hydroDL import master, utils
from hydroDL.master import default
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat


columns=[]
for i in range(41):
    a = str(1980+i)
    columns.append(a)
print(columns)
print(len(columns))

path1 = os.path.join('G:\\Farshid\\SNTemp_newData\\example\\scratch\\SNTemp\\T_S_1713_GAGESII.feather')
inputdata = pd.read_feather(path1)
site_no = inputdata['site_no'].unique()
ValidData_T = pd.DataFrame(0, index=site_no, columns=columns)  # T means water temperature
ValidData_S = pd.DataFrame(0, index=site_no, columns=columns)  # S means stream flow


# finding the valid data for streamflow

for i, ii in enumerate(site_no):
    for j in range(40):
        starttime = str(1980+j)+'-'+'10-01'
        endtime = str(1981+j)+'-'+'10-01'
        A = inputdata.loc[(inputdata['site_no']==ii)
                  & (inputdata['datetime']>= starttime)
                  & (inputdata['datetime']< endtime)
                  & (np.isnan(inputdata['00060_Mean'])==False)]
        ValidData_S.iloc[i,j] = len(A)
    print(i)
path2 = os.path.join('G:\\Farshid\\SNTemp_newData\\example\\scratch\\SNTemp\\validdata_S_pycharm.csv')
ValidData_S.to_csv(path2)


