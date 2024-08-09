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
import numpy as np
import scipy as sp
import eelbrain as eb
import matplotlib
import matplotlib.pyplot as plt
import glob
import cochlea
import time

from localCode.freqAnalysis import *

matplotlib.use('QtAgg')
plt.ion()

# %load_ext autoreload
# %autoreload 2

# %%
#regressorDir='/Users/karl/Dropbox/UMD/ffrTests/'

#mainDirs=['/Users/karl/map/stimAndPredictors/targets/','/Users/karl/map/stimAndPredictors/distractors/','/Users/karl/map/stimAndPredictors/mixes/']
#regTypes=['target','distractor','mix']

mainDirs=['/Users/karl/map/stimAndPredictors/targets/','/Users/karl/map/stimAndPredictors/distractors/','/Users/karl/map/stimAndPredictors/mixes/']
regTypes=['target','distractor','mix']
conditions=['A','B','C','D']

# mainDirs=['/Users/karl/map/stimAndPredictors/mixes/']
# regTypes=['mix']
# conditions=['D']
skipDist=np.array([[2,5,10,13],[2,5,8,11],[2,5,10,15],[2,5,7,12]])  # Skip these distractors, because these trials have no distractor

#matFileNames=['longTrialNoTrigger_forReg_1.mat','longTrialNoTrigger_forReg_2.mat','longTrialNoTrigger_forReg_3.mat','longTrialNoTrigger_forReg_4.mat','longTrialNoTrigger_forReg_5.mat']
#matFieldName='currentAllTrials'

anf_types=['hsr']  # select lsr, msr, or hsr for spont rate

# cfs=(125, 20000, 100)  # This was in the Cochlea repository example
# cfs=(125, 6000, 50)  # 43 comes from the scripting I inherited from Maddox/Vrishab
cfs=(125, 16000, 43)  # 43 comes from the scripting I inherited from Maddox/Vrishab

fsForModel=100000  # This is the sampling rate that the Zilany/Carney model needs to run it, so upsample if we don't have that
fsForRegressor=5000  # This is the sampling rate we want for the regressor eventually, so downsample at the end of everything

# %%
for i, mainDir in enumerate(mainDirs):
    
    for j, condition in enumerate(conditions):
        
        for k in np.arange(1,17):

            ### Make sound
            #fs = 100e3
            #t = np.arange(0, 0.1, 1/fs)
            #s = dsp.chirp(t, 80, t[-1], 20000)
            #s = cochlea.set_dbspl(s, 50)
            #s = np.concatenate( (s, np.zeros(int(10e-3 * fs))) )

            t0=time.time()

            #stimulus=sp.io.loadmat(f'{regressorDir}{matFileName}')[matFieldName]
            
            wavFileName=f'{condition}_{regTypes[i]}{k}.wav'
            
            fs,stimulus=sp.io.wavfile.read(mainDir+wavFileName)
            print(f'Wav file has been read in with sampling rate of {fs} resulting in stimulus of shape {stimulus.shape}')
            print('\n')
            #stimulus=stimulus.squeeze()
            #print(stimulus.shape)
            #fs=sp.io.loadmat(f'{regressorDir}{matFileName}')['fs']
            stimulus=sp.signal.resample(stimulus,int(len(stimulus)*fsForModel/fs))
            stimulus=cochlea.set_dbspl(stimulus,70)
            print(f'Stimulus has been resampled and normalized resulting in shape {stimulus.shape}')
            print('\n')

            #if regTypes[i]=='distractor' and not np.any(skipDist[j]==k):
            if not np.any(skipDist[j]==k) or regTypes[i]!='distractor':

                ### Run model
                rates = cochlea.run_zilany2014_rate(
                    stimulus,
                    fsForModel,
                    anf_types=anf_types,
                    cf=cfs,
                    powerlaw='approximate',
                    species='human'
                )

                rates=rates.to_numpy()

                rates=rates.mean(axis=1)

                rates=sp.signal.resample(rates,int(len(rates)*fsForRegressor/fsForModel))

                t1=time.time()

                print(f'AN model took {t1-t0} seconds to run')
                print(f'AN model to be saved is shape {rates.shape}')
                print('\n')

                eb.save.pickle(rates,f'{mainDir}predictors/{wavFileName[:-4]}_ANmodel.pickle')



                print(f'Done creating and saving AN model for {wavFileName}')
                print('\n')
                print('\n')


                ### Plot rates
                #fig, ax = plt.subplots()
                #img = ax.imshow(
                #    rates.T,
                #    aspect='auto'
                #)
                #plt.colorbar(img)
                #plt.show()

# %%
#print(type(rates))
#print(rates.shape)

# %%
#t=np.linspace(0,int(len(rates)/fsForRegressor),len(rates))
#plt.figure()
#plt.plot(t,rates,t,1000*sp.signal.resample(stimulus,len(rates)))

# %%
#test=eb.load.unpickle('/Users/karl/map/stimAndPredictors/targets/predictors/C_target13~gammatone-1.pickle')

# %%
#test.x.shape

# %%
#plt.figure()
#plt.plot(test.x)

# %%

# %%
