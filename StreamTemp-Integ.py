import sys
sys.path.append('../')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from hydroDL import master, utils
from hydroDL.master import default
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat


import numpy as np
import os
import torch
import pandas as pd
import math
import random

# forcing_list = ['forcing_60%_days_306sites.feather'
#                 ]
# attr_list = ['attr_temp60%_days_306sites.feather'
#              ]


# Options for different interface
interfaceOpt = 1
# ==1 is the more interpretable version, explicitly load data, model and loss, and train the model.
# ==0 is the "pro" version, efficiently train different models based on the defined dictionary variables.
# the results are identical.

# Options for training and testing
# 0: train base model

# 2: test trained models
Action = [0, 2]
# Set hyperparameters for training or retraining
EPOCH = 30
BATCH_SIZE = 59
RHO = 365
HIDDENSIZE = 100
saveEPOCH = 10   # it was 50 then 100
Ttrain = [20101001, 20141001]  # Training period. it was [19851001, 19951001]
seed = None   # fixing the random seed. None means it is not fixed

###############################
TempTarget = '00010_Mean'   # 'obs' or     or Q9Tw   '00010_Mean'  outlet_tave_water


absRoot = os.getcwd()


# Define root directory of database and output
# Modify this based on your own location
rootDatabase = os.path.join(os.path.sep, absRoot, 'scratch', 'SNTemp')  #  dataset root directory:
rootOut = os.path.join(os.path.sep, absRoot, 'TempDemo', 'FirstRun')  # Model output root directory:

forcing_path = os.path.join(os.path.sep, rootDatabase, 'Forcing', 'Forcing_new', 'no_dam_forcing_60%_days118sites.feather')  #
forcing_data =[]#pd.read_feather(forcing_path)
attr_path = os.path.join(os.path.sep, rootDatabase, 'Forcing', 'attr_new', 'no_dam_attr_temp60%_days118sites.feather')
attr_data =[]#pd.read_feather(attr_path)
camels.initcamels(forcing_data, attr_data, TempTarget, rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict



# Define all the configurations into dictionary variables
# three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
# be used to directly train the model when interfaceOpt == 0
# define dataset
optData = default.optDataCamels
optData = default.update(optData, tRange=Ttrain, target='StreamTemp', doNorm=[True,True])  # Update the training period
# define model and update parameters
if torch.cuda.is_available():
    optModel = default.optLstm
else:
    optModel = default.update(
        default.optLstm,
        name='hydroDL.model.rnn.CpuLstmModel')
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
# define loss function
optLoss = default.optLossRMSE
# define training options
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seed)
# define output folder for model results
exp_name = 'TempDemo'
exp_disp = 'FirstRun'


save_path = os.path.join(absRoot, exp_name, exp_disp, \
            'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                          optTrain['miniBatch'][1],
                                                                          optModel['hiddenSize'],
                                                                          optData['tRange'][0], optData['tRange'][1]))
out = os.path.join(rootOut, save_path, 'All-2010-2016') # output folder to save results




##############################################################




# Wrap up all the training configurations to one dictionary in order to save into "out" folder
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

# Train the base model without data integration
if 0 in Action:
    if interfaceOpt == 1:  # use the more interpretable version interface

        #fixing random seeds
        if optTrain['seed'] is None:
            # generate random seed
            randomseed = int(np.random.uniform(low=0, high=1e6))
            optTrain['seed'] = randomseed
            print('random seed updated!')
        else:
            randomseed = optTrain['seed']

        random.seed(randomseed)
        torch.manual_seed(randomseed)
        np.random.seed(randomseed)
        torch.cuda.manual_seed(randomseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



        # load data
        df, x, y, c = master.loadData(optData, TempTarget, forcing_path, attr_path, out)  # df: CAMELS dataframe; x: forcings; y: streamflow obs; c:attributes
        # main outputs of this step are numpy ndArrays: x[nb,nt,nx], y[nb,nt, ny], c[nb,nc]
        # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
        # ny: number of target variables, nc: number of constant attributes
        nx = x.shape[-1] + c.shape[-1]  # update nx, nx = nx + nc
        ny = y.shape[-1]

        if torch.cuda.is_available():
            model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        else:
            model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)

        optModel = default.update(optModel, nx=nx, ny=ny)
        # the loaded model should be consistent with the 'name' in optModel Dict above for logging purpose
        lossFun = crit.RmseLoss()
        # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose
        # update and write the dictionary variable to out folder for logging and future testing
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # train model

        out1 = out


        ############
        model = train.trainModel(
            model,
            x,
            y,
            c,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out)
    elif interfaceOpt==0: # directly train the model using dictionary variable
        master.train(masterDict)




# Test models
if 2 in Action:
    TestEPOCH = 30     # it was 200  # choose the model to test after trained "TestEPOCH" epoches
    # generate a folder name list containing all the tested model output folders
    caseLst = ['All-2010-2016']#, '494-B247-H100','460-B230-H100' ,'327-B163-H100','258-B129-H100' ,'169-B169-H100', '29-B29-H100']

    nDayLst = [] #[1, 3]
    for nDay in nDayLst:
        caseLst.append('All-85-95-DI' + str(nDay))

    outLst = [os.path.join(rootOut, save_path, x) for x in caseLst]
    subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
    tRange = [20141001, 20161001]  # Testing period
    predLst = list()
    obsLst = list()
    predLst_res = list()
    obsLst_res = list()
    statDictLst = []
    for i, out in enumerate(outLst):
        #df, pred, obs = master.test(out, TempTarget, forcing_path[i], attr_path[i], tRange=tRange, subset=subset, basinnorm=True, epoch=TestEPOCH, reTest=True)
        df, pred, obs, x = master.test(out, TempTarget, forcing_path, attr_path, tRange=tRange, subset=subset, basinnorm=False, epoch=TestEPOCH, reTest=True)

        # change the units ft3/s to m3/s
        #obs = obs * 0.0283168
        #pred = pred * 0.0283168
        predLst.append(pred) # the prediction list for all the models
        obsLst.append(obs)
        np.save(os.path.join(out, 'pred.npy'), pred)
        np.save(os.path.join(out, 'obs.npy'), obs)
        f = np.load(os.path.join(out, 'x.npy'))  # it has been saved previously in the out directory (forcings)
        T = (f[:, :, 3] + f[:, :, 4]) / 2    # mean air T for T_residual
        T_air = np.expand_dims(T, axis=2)
        pred_res = pred - T_air
        obs_res = obs - T_air
        predLst_res.append(pred_res)
        obsLst_res.append(obs_res)
    # calculate statistic metrics
       # statDict = stat.statError(pred.squeeze(), obs.squeeze())
      #  statDictLst.append([statDict])
#    statDictLst1 = [stat.statError(x.squeeze(), obs.squeeze()) for x, y in predLst]
    statDictLst = [stat.statError(x.squeeze(), y.squeeze()) for (x, y) in zip(predLst, obsLst)]
    statDictLst_res = [stat.statError_res(x.squeeze(), y.squeeze(), z.squeeze(), w.squeeze()) for (x, y, z, w) in
                   zip(predLst, obsLst, predLst_res, obsLst_res)]

    # median and STD calculation
    count = 0
    mdstd = np.zeros([len(statDictLst_res[0]),3])
    for i in statDictLst_res[0].values():
        median = np.nanmedian((i))    # abs(i)
        STD = np.nanstd((i))        # abs(i)
        mean = np.nanmean((i))      #abs(i)
        k = np.array([[median,STD, mean]])
        mdstd[count] = k
        count = count +1
    mdstd = pd.DataFrame(mdstd, index=statDictLst_res[0].keys(), columns=['median', 'STD','mean'])

    mdstd.to_csv((os.path.join(rootOut, save_path, "mdstd.csv")))




    # Show boxplots of the results
    plt.rcParams['font.size'] = 14
    keyLst = ['Bias','RMSE','ubRMSE', 'NSE', 'Corr', 'NSE_res', 'Corr_res']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst_res)):
            data = statDictLst_res[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)
    labelname =['20%pub', '1%_data', '20%_data', '70%_data', '90%_data', '98%_data', '100%_data']#['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']
    for nDay in nDayLst:
        labelname.append('DI(' + str(nDay) + ')')
    xlabel = ['Bias ($\mathregular{deg}$C)','RMSE', 'ubRMSE', 'NSE', 'Corr', 'NSE_res', 'Corr_res']
    fig = plot.plotBoxFig(dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8))
    fig.patch.set_facecolor('white')
    boxPlotName = "Target:"+TempTarget+" ,epochs="+str(TestEPOCH)+" ,Hiddensize="+str(HIDDENSIZE)+" ,RHO="+str(RHO)+" ,Batches="+str(BATCH_SIZE)
    fig.suptitle(boxPlotName, fontsize=12)
    plt.rcParams['font.size'] = 12

    plt.savefig(os.path.join(rootOut, save_path, (str(len(forcing_path))+"Boxplot.png")))   #, dpi=500
    fig.show()


    # Plot timeseries and locations

    gageindex=[0]
    t = utils.time.tRange2Array(tRange)

    attr = pd.read_feather(attr_path)


    #plot.TempSeries_4_Plots_ERL(attr_path, statDictLst_res, obs, predLst, TempTarget, tRange, boxPlotName, rootOut, save_path, sites=18, Stations=None)

    plot.plotMap(statDictLst[0]['NSE'], lat=attr['lat'].to_numpy(), lon=attr['lon'].to_numpy(), title='NSE'+boxPlotName)


    plt.savefig((os.path.join(rootOut, save_path, "MapNSE-LowRes.png")), bbox_inches='tight')
    plt.show()

