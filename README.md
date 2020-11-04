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
sb.login(username, password) # enter your username and password. this is only necessary before the data are made public
os.mkdir('datarelease')
sb.get_item_files(sb.get_item('5f908db182ce720ee2d0fef9'), 'datarelease') # gage locations
sb.get_item_files(sb.get_item('5f986594d34e198cb77ff084'), 'datarelease') # temperature observations
sb.get_item_files(sb.get_item('5f9865abd34e198cb77ff086'), 'datarelease') # model drivers and basin attributes
sb.get_item_files(sb.get_item('5f9865e5d34e198cb77ff08a'), 'datarelease') # temperature predictions

# Extract the zipfiles
ZipFile('datarelease/01_gage_locations.zip', 'r').extractall('datarelease')
ZipFile('datarelease/weather_drivers.zip', 'r').extractall('datarelease')
```

2. Convert data files from csv to pandas:
```python
# [in python]
import pandas as pd
import shapefile
import numpy as np

# read and combine the basin coordinates and attributes files; save as feather
coords_shp = shapefile.Reader('datarelease/gage_locations.shp')
coords = pd.DataFrame(coords_shp.records(), columns=['site_no', 'site_name', 'lat', 'long']).\
    rename(columns={'long': 'lon'})
attr = pd.read_csv('datarelease/AT_basin_attributes.csv', dtype={'site_no': 'str'}).\
    merge(coords, how='outer')
attr['site_no'] = pd.to_numeric(attr['site_no'])
attr.to_feather('input/no_dam_attr_temp60%_days118sites.feather')

# read the component forcing files
weather = pd.read_csv('datarelease/weather_drivers.csv', parse_dates = ['datetime'])
wtemp = pd.read_csv('datarelease/temperature_observations.csv', parse_dates = ['datetime']).\
    rename(columns={'wtemp(C)': '00010_Mean'})
discharge = pd.read_csv('datarelease/obs_discharge.csv', parse_dates = ['datetime']).\
    rename(columns={'discharge(cfs)': '00060_Mean'})
sim_discharge = pd.read_csv('datarelease/pred_discharge.csv', parse_dates = ['datetime'])

# combine forcings into a single file and save
forcings = pd.merge(weather, wtemp, how='outer').\
    merge(discharge, how='outer').\
    merge(sim_discharge, how='left')
forcings['combine_discharge'] = np.where(
    forcings['datetime'] >= np.datetime64('2014-10-01'),
    forcings['sim_discharge(cfs)'], forcings['00060_Mean']) # combine_discharge has observed Q to train, sim Q to test
forcings.to_feather('input/no_dam_forcing_60%_days118sites.feather')
```

3. Edit lines 19-21: in hydroDL/data/camels.py to set the forcing and basin attribute variables appropriate to the model of interest.

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

When switching between the above model options, delete input/Statistics_basinnorm.json before running the next model.

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

## Environment used in manuscript

The python environment used to generate the precise outputs in this model was:
```
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
