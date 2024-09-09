# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: neurotools
# ---

# %%
import mne
import numpy as np
import eelbrain as eb
import scipy as sp
import mat73
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pickle
import glob
import time
import gc

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *

matplotlib.use("QtAgg")
plt.ion()

# %%
parentDir = "/Users/karl/map/"
# parentDir = "/Volumes/Seagate/map/"

subDirs = "/eegAndMeg/eeg/"
nChannels = 32

# %%
nameOfRegressor = "_ANmodel_correctedLevels"
lenResponse = 425

# nameOfRegressor = "~gammatone-1"
# lenResponse = 1100

# nameOfRegressor = "~gammatone-on-1"
# lenResponse = 1100


# typeOfRegressor = "mix"
typeOfRegressor = "target"
# typeOfRegressor = "distractor"

# timeToAddToStart = 0
timeToAddToStart = 0.008


subjectsToAverage = [
    "R3045",
    "R3089",
    "R3093",
    "R3095",
    "R3151",
    "R2774",
    "R3152",
    "R2877",
    "R3157",
    "R2783",
]

# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877", "R3157"]
# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877"]
# subjectsToAverage=["R3095", "R3151", "R2774", "R3152", "R2877", "R3157", "R2783"]
# subjectsToAverage=["R2877", "R3151", "R3152"]
# subjectsToAverage=["R3152"]


avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

for i, subject in enumerate(subjectsToAverage):
    eegLocation = parentDir + subject + subDirs
    evoked = mne.read_evokeds(
        f"{eegLocation}evoked{nameOfRegressor}_{typeOfRegressor}-ave.fif"
    )
    evoked = evoked[0]
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat

evokedAvg = mne.EvokedArray(
    avgMat.mean(axis=2).T, evoked.info, tmin=evoked.times[0], comment="Target"
)

# %%
# evokedAvg.pick_types(eeg=True).plot_topo(color="r", legend=False)

# %%
# typeOfRegressor = "mix"
# typeOfRegressor = "target"
typeOfRegressor = "distractor"


avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

for i, subject in enumerate(subjectsToAverage):
    eegLocation = parentDir + subject + subDirs
    evoked = mne.read_evokeds(
        f"{eegLocation}evoked{nameOfRegressor}_{typeOfRegressor}-ave.fif"
    )
    evoked = evoked[0]
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat

evokedAvg2 = mne.EvokedArray(
    avgMat.mean(axis=2).T, evoked.info, tmin=evoked.times[0], comment="Distractor"
)

# %%
typeOfRegressor = "mix"
# typeOfRegressor = "target"
# typeOfRegressor = "distractor"


avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

for i, subject in enumerate(subjectsToAverage):
    eegLocation = parentDir + subject + subDirs
    evoked = mne.read_evokeds(
        f"{eegLocation}evoked{nameOfRegressor}_{typeOfRegressor}-ave.fif"
    )
    evoked = evoked[0]
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat

evokedAvg3 = mne.EvokedArray(
    avgMat.mean(axis=2).T, evoked.info, tmin=evoked.times[0], comment="Mix"
)

# %%
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2, evokedAvg3], color=["blue", "red", "green"], legend=True)
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2], color=["blue", "red"], legend=True)
mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2, evokedAvg3], legend=True)
mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2], legend=True)


# %%
