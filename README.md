## LSTM Temperature Model

This README explains how to use the provided data and code in this data release to replicate the training and prediction steps for the stream temperature models (LSTM and LR) described in the paper.

### Configure software environment

Create and activate the conda environment with all necessary software packages. Choose the `condaenv_lstm_xx.yml` file that fits your operating system, or create your own using code from the "Conda environment preparation" section below
```shell script
# [in a shell]
conda update -n base -c defaults conda
conda env create -f condaenv_lstm_linux.yml -n lstm_tq
conda activate lstm_tq
```

### Prepare data

Steps to acquire the data to run this code:

1. Download files from ScienceBase. This can be done using the ScienceBase browser interface or using the `sciencebasepy` python package as follows:
```python
# [in python]
import os
import sciencebasepy
from zipfile import ZipFile
sb = sciencebasepy.SbSession()
sb.login(username, password) # enter your username and password
os.mkdir('scratch/datarelease')
sb.get_item_files(sb.get_item('5f986594d34e198cb77ff084'), 'scratch/datarelease') # temperature and flow observations
sb.get_item_files(sb.get_item('5f9865abd34e198cb77ff086'), 'scratch/datarelease') # model drivers and basin attributes
sb.get_item_files(sb.get_item('5f9865e5d34e198cb77ff08a'), 'scratch/datarelease') # flow predictions

# Extract the zipfiles
ZipFile('scratch/datarelease/weather_drivers.zip', 'r').extractall('scratch/datarelease')
```

2. Convert data files from csv to pandas:
```python
# [in python]
import pandas as pd
import numpy as np

# read the attributes csv file and save as feather
attr = pd.read_csv('scratch/datarelease/AT_basin_attributes.csv')
attr.to_feather('scratch/SNTemp/Forcing/attr_new/no_dam_attr_temp60%_days118sites.feather')

# read the component forcing files
weather = pd.read_csv('scratch/datarelease/weather_drivers.csv', parse_dates = ['datetime'])
wtemp = pd.read_csv('scratch/datarelease/temperature_observations.csv', parse_dates = ['datetime']).\
    rename(columns={'temp_degC': '00010_Mean'})
discharge = pd.read_csv('scratch/datarelease/flow_observations.csv', parse_dates = ['datetime']).\
    rename(columns={'discharge_cfs': '00060_Mean'})
sim_discharge = pd.read_csv('scratch/datarelease/pred_discharge.csv', parse_dates = ['datetime'])

# combine forcings into a single file and save
forcings = pd.merge(weather, wtemp, how='outer').merge(discharge, how='outer').merge(sim_discharge, how='left')
forcings['combine_discharge'] = np.where(
    forcings['datetime'] >= np.datetime64('2014-10-01'),
    forcings['sim_discharge_cfs'], forcings['00060_Mean'])
forcings.to_feather('scratch/SNTemp/Forcing/Forcing_new/no_dam_forcing_60%_days118sites.feather')

#attrf = pd.read_feather('scratch/no_dam_attr_temp60%_days118sites.feather')
#forcingsf = pd.read_feather('scratch/no_dam_forcing_60%_days118sites.feather')

attro = pd.read_csv('../rahmani_erl_data_release/in_data/Data - ERL paper/Forcing_attrFiles/no_dam_attr_temp60__days118sites.csv')
forcingso = pd.read_csv('../rahmani_erl_data_release/in_data/Data - ERL paper/Forcing_attrFiles/no_dam_forcing_60__days118sites.csv', parse_dates=['datetime'])
forcingso[(forcingso['datetime'] >='2010-10-01') & (forcingso['datetime'] <= '2016-09-30')]

```

I'm currently getting this error when I try to use the reprocessed data:
```shell script
# [in a shell]
$ python StreamTemp-Integ.py 
loading package hydroDL
random seed updated!
Traceback (most recent call last):
  File "StreamTemp-Integ.py", line 129, in <module>
    df, x, y, c = master.loadData(optData, TempTarget, forcing_path, attr_path, out)  # df: CAMELS dataframe; x: forcings; y: streamflow obs; c:attributes
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/master/master.py", line 186, in loadData
    rmNan=optData['rmNan'][0])
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/data/camels.py", line 528, in getDataTs
    x[k, :, :] = data
ValueError: could not broadcast input array from shape (10286,7) into shape (14610,7)
```

Same error after deleting some comments and setting `tRange = tRangeobs = [20100101, 20161001]` instead of `[19800101, 20200101]`:
```
Traceback (most recent call last):
  File "StreamTemp-Integ.py", line 129, in <module>
    df, x, y, c = master.loadData(optData, TempTarget, forcing_path, attr_path, out)  # df: CAMELS dataframe; x: forcings; y: streamflow obs; c:attributes
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/master/master.py", line 186, in loadData
    rmNan=optData['rmNan'][0])
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/data/camels.py", line 493, in getDataTs
    x[k, :, :] = data
ValueError: could not broadcast input array from shape (2296,7) into shape (2465,7)
```
and after correcting `20100101` to `20101001` I no longer hit that error, now get
```
Traceback (most recent call last):
  File "StreamTemp-Integ.py", line 137, in <module>
    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/model/rnn.py", line 361, in __init__
    inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)          # for LSTM-untied:    inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/model/rnn.py", line 262, in __init__
    self.cuda()
  File "/home/aappling/miniconda3/envs/lstm_tq/lib/python3.6/site-packages/torch/nn/modules/module.py", line 311, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/caldera/projects/usgs/water/iidd/datasci/psu/LSTM_Temperature/hydroDL/model/rnn.py", line 268, in _apply
    ret = super(CudnnLstm, self)._apply(fn)
  File "/home/aappling/miniconda3/envs/lstm_tq/lib/python3.6/site-packages/torch/nn/modules/module.py", line 230, in _apply
    param_applied = fn(param)
  File "/home/aappling/miniconda3/envs/lstm_tq/lib/python3.6/site-packages/torch/nn/modules/module.py", line 311, in <lambda>
    return self._apply(lambda t: t.cuda(device))
RuntimeError: CUDA error: out of memory
```

3. Edit lines 36-48 in hydroDL/data/camels.py to set the forcing and basin attribute variables appropriate to the model of interest.

For Ts,obsQ:
```python
forcingLst = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', '00060_Mean']
```
For Ts,noQ:
```python
forcingLst = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
```
For Ts,simQ:
```python
forcingLst = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'combine_discharge']
```

For all three models:
```python
attrLstSel = [
    'DRAIN_SQKM', 'STREAMS_KM_SQ_KM', 'STOR_NID_2009', 'FORESTNLCD06', 'PLANTNLCD06', 'SLOPE_PCT',
    'RAW_DIS_NEAREST_MAJ_DAM', 'PERDUN', 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS', 'T_MIN_BASIN',
    'T_MINSTD_BASIN', 'RH_BASIN', 'RAW_AVG_DIS_ALLDAMS', 'PPTAVG_BASIN', 'HIRES_LENTIC_PCT', 'T_AVG_BASIN',
    'T_MAX_BASIN','T_MAXSTD_BASIN', 'NDAMS_2009', 'ELEV_MEAN_M_BASIN']
```

In contrast, the discharge model would have used:
```python
# attrLstSel = ['DRAIN_SQKM', 'PPTAVG_BASIN', 'T_AVG_BASIN', 'T_MAX_BASIN',
#        'T_MAXSTD_BASIN', 'T_MIN_BASIN', 'T_MINSTD_BASIN', 'RH_BASIN',
#        'STREAMS_KM_SQ_KM', 'PERDUN', 'HIRES_LENTIC_PCT', 'NDAMS_2009',
#        'STOR_NID_2009', 'FORESTNLCD06', 'PLANTNLCD06', 'ELEV_MEAN_M_BASIN',
#        'SLOPE_PCT', 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS',
#        'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS',
#        'MAJ_NDAMS_2009', 'POWER_NUM_PTS', 'POWER_SUM_MW', 'lat', 'lon',
#        'HYDRO_DISTURB_INDX', 'BFI_AVE', 'FRAGUN_BASIN', 'DEVNLCD06',
#        'PERMAVE', 'RFACT', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
#        'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06',
#        'EMERGWETNLCD06', 'GEOL_REEDBUSH_DOM_PCT',
#        'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
#        'ARTIFPATH_MAINSTEM_PCT', 'PERHOR', 'TOPWET', 'CONTACT', 'CANALS_PCT',
#        'RAW_AVG_DIS_ALLCANALS', 'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES',
#        'FRESHW_WITHDRAWAL', 'PCT_IRRIG_AG', 'ROADS_KM_SQ_KM',
#        'PADCAT1_PCT_BASIN', 'PADCAT2_PCT_BASIN']
```
or possibly
```python
############# Streamflow prediction for CONUS scale  ##########################
# attrLstSel = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM',
#       'HYDRO_DISTURB_INDX', 'STREAMS_KM_SQ_KM', 'BFI_AVE', 'NDAMS_2009',
#       'STOR_NID_2009', 'RAW_DIS_NEAREST_DAM', 'FRAGUN_BASIN', 'DEVNLCD06',
#       'FORESTNLCD06', 'PLANTNLCD06', 'PERMAVE', 'RFACT',
#       'PPTAVG_BASIN', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
#       'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06',
#       'EMERGWETNLCD06', 'GEOL_REEDBUSH_DOM_PCT',
#        'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
#       'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'PERDUN', 'PERHOR',
#       'TOPWET', 'CONTACT', 'CANALS_PCT', 'RAW_AVG_DIS_ALLCANALS',
#        'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES',
#       'RAW_AVG_DIS_ALL_MAJ_DAMS', 'FRESHW_WITHDRAWAL', 'PCT_IRRIG_AG',
#       'POWER_NUM_PTS', 'POWER_SUM_MW', 'ROADS_KM_SQ_KM', 'PADCAT1_PCT_BASIN',
#       'PADCAT2_PCT_BASIN']   # 'GEOL_REEDBUSH_SITE', , 'AWCAVE'
##############################################################################
```

4. Run the following code in an `sh` (e.g., `bash`) terminal to train the LSTM and predict stream temperature.
A GPU and 256GB of memory are strongly recommended.
```sh
# in bash or similar:
python StreamTemp-Integ.py
```

5. Extract results. 


## Conda environment preparation

You may not need to run this yourself, but here's how the conda environment YAML was prepared:
```sh
# update and configure conda
conda update -n base -c defaults conda
conda config --set channel_priority flexible
conda config --prepend channels conda-forge
conda config --prepend channels defaults
conda config --prepend channels pytorch

# create the environment and install modules:
conda create -n lstm_tq
conda activate lstm_tq
conda install python matplotlib=2.2.0 basemap numpy pandas scipy time statsmodels pyarrow pytorch=1.2.0
pip install sciencebasepy

# export to YAML:
conda env export -n lstm_tq | grep -v "^prefix: " > condaenv_lstm_linux.yml
```

Current error with EPOCH = saveEPOCH = TestEPOCH = 5 instead of 2000:
```
loading package hydroDL
random seed updated!
Local calibration kernel is shut down!
write master file /Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/TempDemo/FirstRun/epochs5_batch59_rho365_hiddensize100_Tstart20101001_Tend20141001/All-2010-2016/master.json
/Users/aappling/opt/anaconda3/envs/lstm_tq/lib/python3.6/site-packages/torch/nn/functional.py:1350: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
  warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
/Users/aappling/opt/anaconda3/envs/lstm_tq/lib/python3.6/site-packages/torch/nn/functional.py:1339: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
Epoch 1 Loss 0.450 time 34.59
Epoch 2 Loss 0.277 time 34.04
Epoch 3 Loss 0.253 time 31.35
Epoch 4 Loss 0.241 time 31.38
Epoch 5 Loss 0.232 time 30.79
read master file /Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/TempDemo/FirstRun/epochs5_batch59_rho365_hiddensize100_Tstart20101001_Tend20141001/All-2010-2016/master.json
read master file /Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/TempDemo/FirstRun/epochs5_batch59_rho365_hiddensize100_Tstart20101001_Tend20141001/All-2010-2016/master.json
output files: ['/Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/TempDemo/FirstRun/epochs5_batch59_rho365_hiddensize100_Tstart20101001_Tend20141001/All-2010-2016/All_20141001_20161001_ep5_Streamflow.csv']
Runing new results
Local calibration kernel is shut down!
Traceback (most recent call last):
  File "/Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/StreamTemp-Integ.py", line 189, in <module>
    df, pred, obs, x = master.test(out, TempTarget, forcing_path, attr_path, tRange=tRange, subset=subset, basinnorm=False, epoch=TestEPOCH, reTest=True)
  File "/Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/hydroDL/master/master.py", line 565, in test
    model, x, c, batchSize=batchSize, filePathLst=filePathLst, doMC=doMC)
  File "/Users/aappling/Documents/Code/Code-PGDL/LSTM_Temperature/hydroDL/model/train.py", line 162, in testModel
    model.train(mode=False)
  File "/Users/aappling/opt/anaconda3/envs/lstm_tq/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1070, in train
    module.train(mode)
TypeError: 'bool' object is not callable
```

## Environment used in manuscript

The python environment 
# LSTM_Temperature Modeling
Using LSTM for stream temperature modeling.
main code is StreamTemp_integ.py.
copy forcing pandas files in scratch/SNTemp/Forcing/Forcing_new
copy attribute pandas files in scratch/SNTemp/Forcing/attr_new

```
# packages in environment :
#
# Name                    Version                   Build  Channel
appdirs                   1.4.4                    pypi_0    pypi
argon2-cffi               20.1.0           py37he774522_1  
attrs                     20.2.0                     py_0  
backcall                  0.2.0                      py_0  
basemap                   1.2.1                    pypi_0    pypi
blas                      1.0                         mkl  
bleach                    3.2.1                      py_0  
ca-certificates           2020.7.22                     0  
certifi                   2020.6.20                py37_0  
cffi                      1.14.3           py37h7a1dbc1_0  
colorama                  0.4.3                      py_0  
cudatoolkit               10.0.130                      0  
cycler                    0.10.0                   py37_0  
dataretrieval             0.4                      pypi_0    pypi
decorator                 4.4.2                      py_0  
defusedxml                0.6.0                      py_0  
distlib                   0.3.1                    pypi_0    pypi
entrypoints               0.3                      py37_0  
filelock                  3.0.12                   pypi_0    pypi
freetype                  2.10.2               hd328e21_0  
icc_rt                    2019.0.0             h0cc432a_1  
icu                       58.2                 ha925a31_3  
importlib-metadata        1.7.0                    py37_0  
importlib_metadata        1.7.0                         0  
intel-openmp              2020.2                      254  
ipykernel                 5.3.4            py37h5ca1d4c_0  
ipython                   7.18.1           py37h5ca1d4c_0  
ipython_genutils          0.2.0                    py37_0  
jedi                      0.17.2                   py37_0  
jinja2                    2.11.2                     py_0  
joblib                    0.17.0                   pypi_0    pypi
jpeg                      9b                   hb83a4c4_2  
jsonschema                3.2.0                    py37_1  
jupyter_client            6.1.6                      py_0  
jupyter_core              4.6.3                    py37_0  
kiwisolver                1.2.0            py37h74a9793_0  
libpng                    1.6.37               h2a8f88b_0  
libsodium                 1.0.18               h62dcd97_0  
libtiff                   4.1.0                h56a325e_1  
lz4-c                     1.9.2                h62dcd97_1  
m2w64-gcc-libgfortran     5.3.0                         6  
m2w64-gcc-libs            5.3.0                         7  
m2w64-gcc-libs-core       5.3.0                         7  
m2w64-gmp                 6.1.0                         2  
m2w64-libwinpthread-git   5.0.0.4634.697f757               2  
markupsafe                1.1.1            py37hfa6e2cd_1  
matplotlib                3.2.2                    pypi_0    pypi
mistune                   0.8.4           py37hfa6e2cd_1001  
mkl                       2020.2                      256  
mkl-service               2.3.0            py37hb782905_0  
mkl_fft                   1.2.0            py37h45dec08_0  
mkl_random                1.1.1            py37h47e9c7a_0  
msys2-conda-epoch         20160418                      1  
nbconvert                 5.6.1                    py37_1  
nbformat                  5.0.7                      py_0  
ninja                     1.10.1           py37h7ef1ec2_0  
notebook                  6.1.1                    py37_0  
numpy                     1.19.1           py37h5510c5b_0  
numpy-base                1.19.1           py37ha3acd2a_0  
olefile                   0.46                     py37_0  
openssl                   1.1.1h               he774522_0    conda-forge
packaging                 20.4                       py_0  
pandas                    1.1.1            py37ha925a31_0  
pandoc                    2.10.1                        0  
pandocfilters             1.4.2                    py37_1  
parso                     0.7.0                      py_0  
patsy                     0.5.1                    py37_0  
permutationimportance     1.2.1.8                  pypi_0    pypi
pickleshare               0.7.5                 py37_1001  
pillow                    7.2.0            py37hcc1f983_0  
pip                       20.2.3                   pypi_0    pypi
pipenv                    2020.8.13                pypi_0    pypi
prometheus_client         0.8.0                      py_0  
prompt-toolkit            3.0.7                      py_0  
pyarrow                   1.0.1                    pypi_0    pypi
pycparser                 2.20                       py_2  
pygments                  2.7.1                      py_0  
pyparsing                 2.4.7                      py_0  
pyproj                    2.6.1.post1              pypi_0    pypi
pyqt                      5.9.2            py37h6538335_2  
pyrsistent                0.17.3           py37he774522_0  
pyshp                     2.1.2                    pypi_0    pypi
python                    3.7.9                h60c2a47_0  
python-dateutil           2.8.1                      py_0  
pytorch                   1.2.0           py3.7_cuda100_cudnn7_1    pytorch
pytz                      2020.1                     py_0  
pywin32                   227              py37he774522_1  
pywinpty                  0.5.7                    py37_0  
pyzmq                     19.0.2           py37ha925a31_1  
qt                        5.9.7            vc14h73c81de_0  
requests                  2.7.0                    pypi_0    pypi
scikit-learn              0.23.2                   pypi_0    pypi
scipy                     1.1.0                    pypi_0    pypi
send2trash                1.5.0                    py37_0  
setuptools                49.6.0                   py37_0  
sip                       4.19.8           py37h6538335_0  
six                       1.15.0                     py_0  
sqlite                    3.33.0               h2a8f88b_0  
statsmodels               0.11.1           py37he774522_0  
terminado                 0.8.3                    py37_0  
testpath                  0.4.4                      py_0  
threadpoolctl             2.1.0                    pypi_0    pypi
tk                        8.6.10               he774522_0  
torchvision               0.4.0                py37_cu100    pytorch
tornado                   6.0.4            py37he774522_1  
tqdm                      4.50.2                   pypi_0    pypi
traitlets                 5.0.4                      py_0  
vc                        14.1                 h0510ff6_4  
virtualenv                20.0.35                  pypi_0    pypi
virtualenv-clone          0.5.4                    pypi_0    pypi
vs2015_runtime            14.16.27012          hf0eaf9b_3  
wcwidth                   0.2.5                      py_0  
webencodings              0.5.1                    py37_1  
wheel                     0.35.1                     py_0  
wincertstore              0.2                      py37_0  
winpty                    0.4.3                         4  
xz                        5.2.5                h62dcd97_0  
zeromq                    4.3.2                ha925a31_3  
zipp                      3.1.0                      py_0  
zlib                      1.2.11               h62dcd97_4  
zstd                      1.4.5                h04227a9_0  
```
Note: you may need to restart the kernel to use updated packages.
