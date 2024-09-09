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
from computeTrfsGeneralized import computeTrfs

matplotlib.use("QtAgg")
plt.ion()

# %load_ext autoreload
# %autoreload 2

# %%
# presStopCorrection = None


# subject = "R3045";badChanList=None;condition="A";presStopCorrection=0
# subject='R3089';badChanList=['P7','CP6','C4','T7','CP5','P3','P4','O2','Oz','PO4'];condition='B';presStopCorrection=0
# subject='R3093';badChanList=['Oz','P8','CP6','Fp2'];condition='C';presStopCorrection=0
# subject='R3095';badChanList=['P7','T8','O2','PO4'];condition='D';presStopCorrection=0  # and possibly O2 and PO4

# subject='R3151';badChanList=['C3','FC5','P4'];condition='A'
# subject='R2774';badChanList=['Fp1','AF3','F7','F3','Fz','F4','FC6','C3','CP5','Pz','CP6','P8','FC1','FC5','T7','AF4'];condition='B'
# subject='R3152';badChanList=['T7','C3','P7','Pz','O1','P8','CP6'];condition='C'
# subject = "R2877"
# # badChanList = ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"]
# badChanList = None
# condition = "D"

# subject='R3157';badChanList=None;condition='A'  # Keep an eye on CP2 and possibly others
# subject='R2783';badChanList=None;condition='B'


subjects = [
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
presStopCorrections = [0, 0, 0, 0, None, None, None, None, None, None]
badChanLists = [
    None,
    ["P7", "CP6", "C4", "T7", "CP5", "P3", "P4", "O2", "Oz", "PO4"],
    ["Oz", "P8", "CP6", "Fp2"],
    ["P7", "T8", "O2", "PO4"],
    ["C3", "FC5", "P4"],
    [
        "Fp1",
        "AF3",
        "F7",
        "F3",
        "Fz",
        "F4",
        "FC6",
        "C3",
        "CP5",
        "Pz",
        "CP6",
        "P8",
        "FC1",
        "FC5",
        "T7",
        "AF4",
    ],
    ["T7", "C3", "P7", "Pz", "O1", "P8", "CP6"],
    ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"],
    None,
    None,
]
conditions = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]


typeOfRegressors = ["mix", "target", "distractor"]


nameOfRegressors = ["_ANmodel_correctedLevels", "~gammatone-1", "~gammatone-on-1"]


regressorDirs = [
    "/Users/karl/map/stimAndPredictors/mixes/predictors/",
    "/Users/karl/map/stimAndPredictors/targets/predictors/",
    "/Users/karl/map/stimAndPredictors/distractors/predictors/",
]

for iSubject, subject in enumerate(subjects):
    for iType, typeOfRegressor in enumerate(typeOfRegressors):
        for nameOfRegressor in nameOfRegressors:

            regressorDir = regressorDirs[iType]

            print("\n")
            print(
                [
                    subject,
                    presStopCorrections[iSubject],
                    badChanLists[iSubject],
                    conditions[iSubject],
                    typeOfRegressor,
                    nameOfRegressor,
                    regressorDir,
                ]
            )
            print("\n")
            computeTrfs(
                subject,
                presStopCorrections[iSubject],
                badChanLists[iSubject],
                conditions[iSubject],
                typeOfRegressor,
                nameOfRegressor,
                regressorDir,
            )
