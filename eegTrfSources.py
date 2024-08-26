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
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
from nilearn import plotting

matplotlib.use("QtAgg")

# %%
# subject = "R3045";badChanList=None;condition="A";presStopCorrection=0
# subject='R3089';badChanList=['P7','CP6','C4','T7','CP5','P3','P4','O2','Oz','PO4'];condition='B';presStopCorrection=0
# subject='R3093';badChanList=['Oz','P8','CP6','Fp2'];condition='C';presStopCorrection=0
# subject='R3095';badChanList=['P7','T8','O2','PO4'];condition='D';presStopCorrection=0  # and possibly O2 and PO4

# subject='R3151';badChanList=['C3','FC5','P4'];condition='A'
# subject='R2774';badChanList=['Fp1','AF3','F7','F3','Fz','F4','FC6','C3','CP5','Pz','CP6','P8','FC1','FC5','T7','AF4'];condition='B'
# subject='R3152';badChanList=['T7','C3','P7','Pz','O1','P8','CP6'];condition='C'
subject = "R2877"
badChanList = ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"]
condition = "D"

# subject='R3157';badChanList=['CP6'];condition='A'  # Keep an eye on CP2 and possibly others
# subject='R3157';badChanList=None;condition='A'  # Keep an eye on CP2 and possibly others
# subject='R2783';badChanList=['T7','P7','CP5'];condition='B'
# subject='R2783';badChanList=None;condition='B'


subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

subDirs = "/eegAndMeg/eeg/"

eegLocation = "/Users/karl/map/" + subject + subDirs


nameOfRegressor = "_ANmodel_correctedLevels"
# nameOfRegressor='_ANmodel'
# nameOfRegressor='~gammatone-1'
# nameOfRegressor='~gammatone-on-1'

baselineFiles = sorted(glob.glob(f"{eegLocation}*baseline*bdf"))
bdfFile = baselineFiles[0]

l_freq = 2
h_freq = None

refs = ["EXG3", "EXG4"]
notchFreqs = np.arange(60, 8192, 60)

labels_vol = [
    "Left-Thalamus",
    "Left-Cerebellum-Cortex",
    "Brain-Stem",
    "Right-Thalamus",
    "Right-Cerebellum-Cortex",
]

evoked = mne.read_evokeds(f"{eegLocation}evoked{nameOfRegressor}-ave.fif")
evoked = evoked[0]
evoked.set_eeg_reference(projection=True)

initial_time = 0.0085

# %%
baselineRaw = mne.io.read_raw_bdf(bdfFile, preload=True)

baselineRaw.set_eeg_reference(ref_channels=refs)
baselineRaw.notch_filter(notchFreqs)

# %%
elp_ch_names = [
    "Mark1",
    "Mark2",
    "Mark3",
    "Mark4",
    "Mark5",
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
]

channelFile = glob.glob("/Users/karl/map/" + subject + "/digitization/*eeg*.elp")
if len(channelFile) == 0:
    channelFile = glob.glob("/Users/karl/map/" + subject + "/digitization/*EEG*.elp")
print(f"Using the channel file {channelFile[0]}")

digMontage = mne.channels.read_dig_polhemus_isotrak(
    channelFile[0], ch_names=elp_ch_names
)
changeChTypes = {
    "EXG1": "eog",
    "EXG2": "eog",
    "EXG3": "eog",
    "EXG4": "eog",
    "EXG5": "eog",
    "EXG6": "eog",
    "EXG7": "eog",
    "EXG8": "eog",
}
elpChangeChTypes = {
    "Mark1": "hpi",
    "Mark2": "hpi",
    "Mark3": "hpi",
    "Mark4": "hpi",
    "Mark5": "hpi",
}

baselineRaw.set_channel_types(changeChTypes)
baselineRaw.set_montage(digMontage)

if badChanList is not None:
    baselineRaw.info["bads"].extend(badChanList)
    baselineRaw.interpolate_bads()

# %%
if l_freq is not None or h_freq is not None:
    # denoised.filter(l_freq=.1,h_freq=20)
    # denoised.filter(l_freq=10,h_freq=50)
    baselineRaw.filter(l_freq=l_freq, h_freq=h_freq)
    print("\n")

# %%
baselineCov = mne.compute_raw_covariance(baselineRaw, picks="eeg")

# %%
surfaces = mne.make_bem_model(subject)
bem = mne.make_bem_solution(surfaces)
# bem = subjects_dir + "/" + subject + "/bem/" + subject + "-head.fif"

trans = eegLocation + "eeg-trans.fif"
mriAseg = subjects_dir + "/" + subject + "/mri/aseg.mgz"

src = mne.setup_source_space(
    subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir
)

vol_src = mne.setup_volume_source_space(
    subject,
    mri=mriAseg,
    pos=10.0,
    bem=bem,
    volume_label=labels_vol,
    subjects_dir=subjects_dir,
    add_interpolator=True,  # just for speed, usually this should be True
    verbose=True,
)

# Generate the mixed source space
src += vol_src
print(
    f"The source space contains {len(src)} spaces and "
    f"{sum(s['nuse'] for s in src)} vertices"
)

# src.plot(subjects_dir=subjects_dir)

# Setup volumn source space (This is the part how to create volSourceEstimate)
# vol_src = mne.setup_volume_source_space(
#     subject, mri=mri, pos=10.0, bem=bem,
#     subjects_dir=subjects_dir,
#     add_interpolator=True,
#     verbose=True)

# %%
fwd = mne.make_forward_solution(
    evoked.info,
    trans,
    src,
    bem,
    mindist=5.0,  # ignore sources<=5mm from innerskull
    meg=False,
    eeg=True,
    n_jobs=None,
)
# del src  # save memory

leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
print(
    f"The fwd source space contains {len(fwd['src'])} spaces and "
    f"{sum(s['nuse'] for s in fwd['src'])} vertices"
)


# %%
snr = 3.0  # use smaller SNR for raw data
inv_method = "dSPM"  # sLORETA, MNE, dSPM
parc = "aparc"  # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
loose = dict(surface=0.2, volume=1.0)
depth = 3

lambda2 = 1.0 / snr**2

inverse_operator = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, baselineCov, depth=depth, loose=loose, verbose=True
)
# del fwd

stc = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, lambda2, inv_method, pick_ori=None
)
src = inverse_operator["src"]

# %%
stc_vec = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, lambda2, inv_method, pick_ori="vector"
)
brain = stc_vec.plot(
    hemi="both",
    src=inverse_operator["src"],
    views="coronal",
    initial_time=initial_time,
    subjects_dir=subjects_dir,
    brain_kwargs=dict(silhouette=True),
    smoothing_steps=7,
)

# %%
brain2 = stc.surface().plot(
    initial_time=initial_time, subjects_dir=subjects_dir, smoothing_steps=7
)

# %%
fig = stc.volume().plot(initial_time=initial_time, src=src, subjects_dir=subjects_dir)

# %%
