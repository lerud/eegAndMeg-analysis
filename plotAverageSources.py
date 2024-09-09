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
plt.ion()


# %%
# brain = stc_vec.plot(
#     hemi="both",
#     src=inverse_operator_mix["src"],
#     views="coronal",
#     initial_time=initial_time,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=7,
#     show_traces=True
# )

# %%
# brain2 = stc.surface().plot(
#     initial_time=initial_time, subjects_dir=subjects_dir, smoothing_steps=7
# )

# %%
# fig = stc.volume().plot(initial_time=initial_time, src=src, subjects_dir=subjects_dir)

# %%

# # Plot electrode locations on scalp
# fig = mne.viz.plot_alignment(
#     evoked.info,
#     trans,
#     subject=mriSubject,
#     dig=True,
#     eeg=["original", "projected"],
#     meg=[],
#     coord_frame="head",
#     subjects_dir=subjects_dir,
#     surfaces=dict(brain=0.4, outer_skull=0.6,inner_skull=0.4, head=None)
# )

# # Set viewing angle
# mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)


# %%
def calcPC(mat, doPC):
    if doPC:
        outmat = np.zeros((mat.shape[2], mat.shape[0]))
        for i in range(mat.shape[0]):
            submat = mat[i, :, :].T
            u, s, _ = np.linalg.svd(submat)
            PC = u[:, 0] * s[0]
            pearsonR = np.corrcoef(PC, submat.mean(axis=1))[1, 0]
            print(pearsonR)
            PC *= np.sign(pearsonR)
            outmat[:, i] = PC
    else:
        outmat = mat.mean(axis=1).T
    return outmat


# %%
doPC = True

nonVecMode = "mean_flip"
doVec = True

# %%
# nameOfRegressor = "_ANmodel_correctedLevels"
# lenResponse = 425

# nameOfRegressor = "~gammatone-1"
# lenResponse = 1100

nameOfRegressor = "~gammatone-on-1"
lenResponse = 1100


# typeOfRegressor = "mix"
typeOfRegressor = "target"
# typeOfRegressor = "distractor"

timeToAddToStart = 0
# timeToAddToStart = .008

subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

subDirs = "/eegAndMeg/eeg/"


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
# subjectsToAverage=["R2877", "R3151"]
# subjectsToAverage=["R3151"]
# subjectsToAverage=["R2877"]
# subjectsToAverage=["R3152"]


audTimeCoursesAllSubjs = []
tempTimeCoursesAllSubjs = []
supParTimeCoursesAllSubjs = []
infParTimeCoursesAllSubjs = []
frontTimeCoursesAllSubjs = []
postcTimeCoursesAllSubjs = []
occTimeCoursesAllSubjs = []
subcortTimeCoursesAllSubjs = []
allSourceTimeCoursesAllSubjs = []


for i, subject in enumerate(subjectsToAverage):

    eegLocation = "/Users/karl/map/" + subject + subDirs
    # eegLocation = "/Volumes/Seagate/map/" + subject + subDirs

    (
        subject,
        mriSubject,
        labels_vol,
        src_surf,
        src_vol,
        src_mix,
        fwd_surf,
        fwd_vol,
        fwd_mix,
        inverse_operator_surf,
        inverse_operator_vol,
        inverse_operator_mix,
        stc,
        stc_vec,
    ) = eb.load.unpickle(
        f"{eegLocation}sources{nameOfRegressor}_{typeOfRegressor}.pickle"
    )

    if doVec:

        allSourceTimeCourses = []

        audLabels = mne.read_labels_from_annot(mriSubject, regexp="transversetemporal")
        audTimeCourses = calcPC(
            stc_vec.extract_label_time_course(audLabels, src_mix)[0:2, :, :], doPC
        )
        print(audTimeCourses.shape)
        allSourceTimeCourses.append(audTimeCourses)

        tempLabels = mne.read_labels_from_annot(mriSubject, regexp="superiortemporal")
        tempTimeCourses = calcPC(
            stc_vec.extract_label_time_course(tempLabels, src_mix)[0:2, :, :], doPC
        )
        print(tempTimeCourses.shape)
        allSourceTimeCourses.append(tempTimeCourses)

        supParLabels = mne.read_labels_from_annot(mriSubject, regexp="superiorparietal")
        supParTimeCourses = calcPC(
            stc_vec.extract_label_time_course(supParLabels, src_mix)[0:2, :, :], doPC
        )
        print(supParTimeCourses.shape)
        allSourceTimeCourses.append(supParTimeCourses)

        infParLabels = mne.read_labels_from_annot(mriSubject, regexp="inferiorparietal")
        infParTimeCourses = calcPC(
            stc_vec.extract_label_time_course(infParLabels, src_mix)[0:2, :, :], doPC
        )
        print(infParTimeCourses.shape)
        allSourceTimeCourses.append(infParTimeCourses)

        frontLabels = mne.read_labels_from_annot(mriSubject, regexp="frontalpole")
        frontTimeCourses = calcPC(
            stc_vec.extract_label_time_course(frontLabels, src_mix)[0:2, :, :], doPC
        )
        print(frontTimeCourses.shape)
        allSourceTimeCourses.append(frontTimeCourses)

        postcLabels = mne.read_labels_from_annot(mriSubject, regexp="postcentral")
        postcTimeCourses = calcPC(
            stc_vec.extract_label_time_course(postcLabels, src_mix)[0:2, :, :], doPC
        )
        print(postcTimeCourses.shape)
        allSourceTimeCourses.append(postcTimeCourses)

        occLabels = mne.read_labels_from_annot(mriSubject, regexp="lateraloccipital")
        occTimeCourses = calcPC(
            stc_vec.extract_label_time_course(occLabels, src_mix)[0:2, :, :], doPC
        )
        print(occTimeCourses.shape)
        allSourceTimeCourses.append(occTimeCourses)

        subcortLabels = []
        subcortTimeCourses = calcPC(
            stc_vec.extract_label_time_course(subcortLabels, src_mix), doPC
        )
        print(subcortTimeCourses.shape)

        allSourceTimeCourses = np.array(allSourceTimeCourses)

        audTimeCoursesAllSubjs.append(audTimeCourses)
        tempTimeCoursesAllSubjs.append(tempTimeCourses)
        supParTimeCoursesAllSubjs.append(supParTimeCourses)
        infParTimeCoursesAllSubjs.append(infParTimeCourses)
        frontTimeCoursesAllSubjs.append(frontTimeCourses)
        postcTimeCoursesAllSubjs.append(postcTimeCourses)
        occTimeCoursesAllSubjs.append(occTimeCourses)
        subcortTimeCoursesAllSubjs.append(subcortTimeCourses)
        allSourceTimeCoursesAllSubjs.append(allSourceTimeCourses)

    else:

        allSourceTimeCourses = []

        audLabels = mne.read_labels_from_annot(mriSubject, regexp="transversetemporal")
        audTimeCourses = stc.extract_label_time_course(
            audLabels, src_surf, mode=nonVecMode
        )
        print(audTimeCourses.shape)
        allSourceTimeCourses.append(audTimeCourses[0:2, :].T)

        tempLabels = mne.read_labels_from_annot(mriSubject, regexp="superiortemporal")
        tempTimeCourses = stc.extract_label_time_course(
            tempLabels, src_surf, mode=nonVecMode
        )
        print(tempTimeCourses.shape)
        allSourceTimeCourses.append(tempTimeCourses[0:2, :].T)

        supParLabels = mne.read_labels_from_annot(mriSubject, regexp="superiorparietal")
        supParTimeCourses = stc.extract_label_time_course(
            supParLabels, src_surf, mode=nonVecMode
        )
        print(supParTimeCourses.shape)
        allSourceTimeCourses.append(supParTimeCourses[0:2, :].T)

        infParLabels = mne.read_labels_from_annot(mriSubject, regexp="inferiorparietal")
        infParTimeCourses = stc.extract_label_time_course(
            infParLabels, src_surf, mode=nonVecMode
        )
        print(infParTimeCourses.shape)
        allSourceTimeCourses.append(infParTimeCourses[0:2, :].T)

        frontLabels = mne.read_labels_from_annot(mriSubject, regexp="frontalpole")
        frontTimeCourses = stc.extract_label_time_course(
            frontLabels, src_surf, mode=nonVecMode
        )
        print(frontTimeCourses.shape)
        allSourceTimeCourses.append(frontTimeCourses[0:2, :].T)

        postcLabels = mne.read_labels_from_annot(mriSubject, regexp="postcentral")
        postcTimeCourses = stc.extract_label_time_course(
            postcLabels, src_surf, mode=nonVecMode
        )
        print(postcTimeCourses.shape)
        allSourceTimeCourses.append(postcTimeCourses[0:2, :].T)

        occLabels = mne.read_labels_from_annot(mriSubject, regexp="lateraloccipital")
        occTimeCourses = stc.extract_label_time_course(
            occLabels, src_surf, mode=nonVecMode
        )
        print(occTimeCourses.shape)
        allSourceTimeCourses.append(occTimeCourses[0:2, :].T)

        subcortLabels = []
        subcortTimeCourses = stc_vec.extract_label_time_course(subcortLabels, src_mix)
        print(subcortTimeCourses.shape)

        allSourceTimeCourses = np.array(allSourceTimeCourses)

        audTimeCoursesAllSubjs.append(audTimeCourses)
        tempTimeCoursesAllSubjs.append(tempTimeCourses)
        supParTimeCoursesAllSubjs.append(supParTimeCourses)
        infParTimeCoursesAllSubjs.append(infParTimeCourses)
        frontTimeCoursesAllSubjs.append(frontTimeCourses)
        postcTimeCoursesAllSubjs.append(postcTimeCourses)
        occTimeCoursesAllSubjs.append(occTimeCourses)
        subcortTimeCoursesAllSubjs.append(subcortTimeCourses)
        allSourceTimeCoursesAllSubjs.append(allSourceTimeCourses)


#         ymax = np.max([subcortTimeCourses.mean(axis=1).T.max(), np.array(allSourceTimeCourses).max()])
#         ymin = np.min([subcortTimeCourses.mean(axis=1).T.min(), np.array(allSourceTimeCourses).min()])

#         figsize = [20,11]
#         fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
#                 nrows=2, ncols=4, figsize=figsize)


#         ax1.plot(stc._times,audTimeCourses[0:2,:].T)
#         ax1.set_ylim([ymin, ymax])
#         ax1.legend([audLabels[0].name, audLabels[1].name])

#         ax2.plot(stc._times,tempTimeCourses[0:2,:].T)
#         ax2.set_ylim([ymin, ymax])
#         ax2.legend([tempLabels[0].name, tempLabels[1].name])

#         ax3.plot(stc._times,supParTimeCourses[0:2,:].T)
#         ax3.set_ylim([ymin, ymax])
#         ax3.legend([supParLabels[0].name, supParLabels[1].name])

#         ax4.plot(stc._times,infParTimeCourses[0:2,:].T)
#         ax4.set_ylim([ymin, ymax])
#         ax4.legend([infParLabels[0].name, infParLabels[1].name])

#         ax5.plot(stc._times,frontTimeCourses[0:2,:].T)
#         ax5.set_ylim([ymin, ymax])
#         ax5.legend([frontLabels[0].name, frontLabels[1].name])

#         ax6.plot(stc._times,postcTimeCourses[0:2,:].T)
#         ax6.set_ylim([ymin, ymax])
#         ax6.legend([postcLabels[0].name, postcLabels[1].name])

#         ax7.plot(stc._times,occTimeCourses[0:2,:].T)
#         ax7.set_ylim([ymin, ymax])
#         ax7.legend([occLabels[0].name, occLabels[1].name])

#         ax8.plot(stc._times,subcortTimeCourses.mean(axis=1).T)
#         # ax8.plot(stc._times,subcortTimeCourses[:,2,:].T)
#         # ax8.set_ylim([ymin, ymax])
#         ax8.legend(labels_vol)

# %%


audTimeCoursesAllSubjs = np.array(audTimeCoursesAllSubjs)
tempTimeCoursesAllSubjs = np.array(tempTimeCoursesAllSubjs)
supParTimeCoursesAllSubjs = np.array(supParTimeCoursesAllSubjs)
infParTimeCoursesAllSubjs = np.array(infParTimeCoursesAllSubjs)
frontTimeCoursesAllSubjs = np.array(frontTimeCoursesAllSubjs)
postcTimeCoursesAllSubjs = np.array(postcTimeCoursesAllSubjs)
occTimeCoursesAllSubjs = np.array(occTimeCoursesAllSubjs)
subcortTimeCoursesAllSubjs = np.array(subcortTimeCoursesAllSubjs)
allSourceTimeCoursesAllSubjs = np.array(allSourceTimeCoursesAllSubjs)


audTimeCoursesAvg = np.mean(audTimeCoursesAllSubjs, axis=0)
tempTimeCoursesAvg = np.mean(tempTimeCoursesAllSubjs, axis=0)
supParTimeCoursesAvg = np.mean(supParTimeCoursesAllSubjs, axis=0)
infParTimeCoursesAvg = np.mean(infParTimeCoursesAllSubjs, axis=0)
frontTimeCoursesAvg = np.mean(frontTimeCoursesAllSubjs, axis=0)
postcTimeCoursesAvg = np.mean(postcTimeCoursesAllSubjs, axis=0)
occTimeCoursesAvg = np.mean(occTimeCoursesAllSubjs, axis=0)
subcortTimeCoursesAvg = np.mean(subcortTimeCoursesAllSubjs, axis=0)
allSourceTimeCoursesAvg = np.mean(allSourceTimeCoursesAllSubjs, axis=0)

# %%
ymax = np.max([subcortTimeCoursesAvg.max(), allSourceTimeCoursesAvg.max()])
ymin = np.min([subcortTimeCoursesAvg.min(), allSourceTimeCoursesAvg.min()])

figsize = [20, 11]

fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
    nrows=2, ncols=4, figsize=figsize
)


ax1.plot(stc._times, audTimeCoursesAvg)
# ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
ax1.set_ylim([ymin, ymax])
ax1.legend([audLabels[0].name, audLabels[1].name])

ax2.plot(stc._times, tempTimeCoursesAvg)
ax2.set_ylim([ymin, ymax])
ax2.legend([tempLabels[0].name, tempLabels[1].name])

ax3.plot(stc._times, supParTimeCoursesAvg)
ax3.set_ylim([ymin, ymax])
ax3.legend([supParLabels[0].name, supParLabels[1].name])

ax4.plot(stc._times, infParTimeCoursesAvg)
ax4.set_ylim([ymin, ymax])
ax4.legend([infParLabels[0].name, infParLabels[1].name])

ax5.plot(stc._times, frontTimeCoursesAvg)
ax5.set_ylim([ymin, ymax])
ax5.legend([frontLabels[0].name, frontLabels[1].name])

ax6.plot(stc._times, postcTimeCoursesAvg)
ax6.set_ylim([ymin, ymax])
ax6.legend([postcLabels[0].name, postcLabels[1].name])

ax7.plot(stc._times, occTimeCoursesAvg)
ax7.set_ylim([ymin, ymax])
ax7.legend([occLabels[0].name, occLabels[1].name])

ax8.plot(stc._times, subcortTimeCoursesAvg)
# ax8.plot(stc._times,subcortTimeCourses[:,1,:].T)
# ax8.set_ylim([ymin, ymax])
ax8.legend(labels_vol)

# %%
