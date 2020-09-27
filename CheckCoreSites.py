import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import stat, plot
import os


def CheckCoreSites(predLst, obsLst, attr_path, CoreSitesLst, out, MainListLabel,
                   boxPlotName= 'Coresites'):
    ### first step: to recognize which row is for which station
    No_Lst = len(predLst)  # shows the number of prediction datasets
    for ii in range(No_Lst):
        attr = pd.read_feather(attr_path)
        mainLst = attr['site_no'].unique()
        CoreLst = CoreSitesLst['site_no'].unique()
        rows = list()
        CorePredLst = list()
        CoreObsLst = list()
        CoreObs = list()
        CorePred = list()
        for jj in range(len(CoreLst)):
            ind = np.where(mainLst == CoreLst[jj])
            rows[jj] = ind[0][0]
            CorePred[jj,:] = predLst[ii][jj]
            CoreObs [jj,:] = obsLst[ii][jj]
        CorePredLst.append(CorePred)
        CoreObsLst.append(CoreObs)
    CorestatDictLst = [stat.statError(x.squeeze(), y.squeeze()) for (x, y) in zip(CorePredLst, CoreObsLst)]
    keyLst = ['Bias', 'RMSE', 'ubRMSE', 'NSE', 'Corr', 'R2']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(CorestatDictLst)):
            data = CorestatDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)

    xlabel = ['Bias ($\mathregular{deg}$C)', 'RMSE', 'ubRMSE', 'NSE', 'Corr', 'R2']
    fig = plot.plotBoxFig(dataBox, xlabel, label2=MainListLabel, sharey=False, figsize=(16, 8))
    fig.patch.set_facecolor('white')
    fig.suptitle(boxPlotName, fontsize=12)
    plt.rcParams['font.size'] = 12
    plt.savefig(os.path.join(out,  boxPlotName))  # , dpi=500
    fig.show()

    Coreattr=pd.DataFrame()
    for jj in range(len(rows)):
        Coreattr.append(attr[rows[jj]], ignore_index=True)
    plot.plotMap(CorestatDictLst[0]['NSE'], lat=Coreattr['lat'].to_numpy(), lon=Coreattr['lon'].to_numpy(),
                 title='NSE' + boxPlotName)
#    plt.savefig(os.path.join(rootOut, save_path, "MapNSE.png"), dpi=500)
    plt.savefig((os.path.join(out, boxPlotName)), bbox_inches='tight')
    plt.show()


