# 2021/09/08~
# Fernando Gama, fgama@rice.edu

################################################################################
#
####                               IMPORTING
#
################################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
import pickle
import datetime
#from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
#import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Modules.particles as particles

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed, loadSeed

# Start measuring time
startRunTime = datetime.datetime.now()

################################################################################
#
####                                SETTING
#
################################################################################

################################################################################
####                          Handling Directories
################################################################################

thisFilename = 'particleFilteringNonlinearSNR' # This is the general name of all related files

########
#### Directory definition
#############################

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

########
#### Directory and file creation
####################################

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Directory to save the specific dataset splits used
dataSplitDir = os.path.join(saveDir, 'dataSplit')
if not os.path.exists(dataSplitDir):
    os.makedirs(dataSplitDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


#\\\ Save seed for reproducibility
saveSeed(saveDir)

################################################################################
#
####                          PARAMETER SETTING
#
################################################################################

################################################################################
####                          Setting Parameters
################################################################################

print("Setting parameters...", end = ' ', flush = True)

useGPU = False # If true, and GPU is available, use it.
useBias = True # If true, make the linear operation be affine

N = 10 # Size of x (state)
M = 8 # Size of y (measurement)
T = 12 # Trajectory length
K = 25
Kthres = K//3 # Threshold for resampling

nDataRealizations = 10 # Times to simulate the data
nSampleRealizations = 100 # Times to simulate the sampling

print("OK")

#########
#### Save
##############

varsToSave = {'useGPU': useGPU,
              'useBias': useBias,
              'N': N,
              'M': M,
              'T': T,
              'K': K,
              'Kthres': Kthres,
              'nDataRealizations': nDataRealizations,
              'nSampleRealizations': nSampleRealizations}

# Save the chosen parameters in the text file
writeVarValues(varsFile, varsToSave)

# Save the corresponding pickle
with open(os.path.join(saveDir, 'settingParameters.pkl'), 'wb') as thisFile:
    pickle.dump(varsToSave, thisFile)

#######################################
####       Matrix parameters
#######################################

print("Setting matrix parameters...", end = ' ', flush = True)

# Graph settings (for a planar graph)
pos = []
graphOptions = {}
graphOptions['pos'] = pos
graphOptions['kernelType'] = 'gaussian'
graphOptions['sparseType'] = 'NN'
graphOptions['sparseParam'] = 3

# And all the non-random parameters
# Matrix C
C = np.eye(M,N) + np.fliplr(np.eye(M,N))

# Distribution values
#   Initial distribution
muo = np.ones(N)#/np.linalg.norm(np.ones(N), ord = 2)
Sigmao = np.eye(N)
#   Noise
SNRstart = 0
SNRend = 10
SNRpoints = 5
SNR = np.linspace(SNRstart, SNRend, SNRpoints)
sigma2 = np.sum(muo ** 2) / (10 ** (SNR/10))
muv = np.zeros(N)
muw = np.zeros(M)

def nonlinearSystem(x): # If we want to create another function
    return x
# f = nonlinearSystem #Nonlinear function, to apply to Ax_{t-1}
f = np.abs

print("OK")

#########
#### Save
##############

varsToSave = {'graphOptions': graphOptions,
              'C': C,
              'muo': muo,
              'Sigmao': Sigmao,
              'SNRstart': SNRstart,
              'SNRend': SNRend,
              'SNRpoints': SNRpoints,
              'SNR': SNR,
              'sigma2': sigma2,
              'muv': muv,
              'muw': muw,
              'f': f}

# Save the chosen parameters in the text file
writeVarValues(varsFile, varsToSave)

# Save the corresponding pickle
with open(os.path.join(saveDir, 'matrixParameters.pkl'), 'wb') as thisFile:
    pickle.dump(varsToSave, thisFile)

#######################################
####      Training parameters
#######################################

print("Setting training parameters...", end = ' ', flush = True)

#\\\ Overall training options
learningRate = 0.001
nEpochs = 200 # Number of epochs
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#########
#### Save
##############

varsToSave = {'learningRate': learningRate,
              'nEpochs': nEpochs,
              'doLearningRateDecay': doLearningRateDecay,
              'learningRateDecayRate': learningRateDecayRate,
              'learningRateDecayPeriod': learningRateDecayPeriod,
              'validationInterval': validationInterval}

writeVarValues(varsFile, varsToSave)

with open(os.path.join(saveDir, 'trainingParameters.pkl'), 'wb') as thisFile:
    pickle.dump(varsToSave, thisFile)

print("OK")

################################################################################
####                     Architecture Hyperparameters
################################################################################

print("Setting model hyperparameters...", end = ' ', flush = True)

modelList = []

doOptmlSIS = True
doRsmplSIS = True
doLearnSIS = True
doLrnRsSIS = True

if doOptmlSIS:
    modelList.append('OptmlSIS')
if doRsmplSIS:
    modelList.append('RsmplSIS')
if doLearnSIS:
    modelList.append('LearnSIS')
if doLrnRsSIS:
    modelList.append('LrnRsSIS')

# The variables for LearnSIS
F = [256, 512]
nonlinearity = nn.Tanh

print("OK")

#########
#### Save
##############

varsToSave = {'doOptmlSIS': doOptmlSIS,
              'doRsmplSIS': doRsmplSIS,
              'doLearnSIS': doLearnSIS,
              'doLrnRsSIS': doLrnRsSIS,
              'modelList': modelList,
              'F': F,
              'nonlinearity': nonlinearity}

# Save the chosen parameters in the text file
writeVarValues(varsFile, varsToSave)

# Save the corresponding pickle
with open(os.path.join(saveDir, 'modelSetting.pkl'), 'wb') as thisFile:
    pickle.dump(varsToSave, thisFile)

################################################################################
####              Logging Parameters (Printing and Figures)
################################################################################

print("Setting logging parameters...", end = ' ', flush = True)

# Options:
doPrint = True # Decide whether to print stuff while running
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)

# Parameters:
computeStatistic = np.median # Choose function to summarize the run's results
printInterval = 0 # After how many training steps, print the partial results

# Figure parameters:
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers
fracErrorBar = 1. # The standard deviation in the error bars is divided by this
fontSize = 22 # Font size
tickSize = 18 # Size of the ticks
legendSize = 18 # Size of legend
trimAxes = False # If true trim the axes (particularly useful if initial losses
    # are too high)
xAxisMultiplierTrain = 1 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.

if doPrint:
    print("OK")
else:
    print("OK -- Not printing anymore")

#########
#### Save
##############

varsToSave = {'doPrint': doPrint,
              'doSaveVars': doSaveVars,
              'doFigs': doFigs,
              'computeStatistic': computeStatistic,
              'printInterval': printInterval,
              'figSize': figSize,
              'lineWidth': lineWidth,
              'markerShape': markerShape,
              'markerSize': markerSize,
              'fracErrorBar': fracErrorBar,
              'fontSize': fontSize,
              'tickSize': tickSize,
              'legendSize': legendSize,
              'trimAxes': trimAxes,
              'xAxisMultiplierTrain': xAxisMultiplierTrain}

writeVarValues(varsFile, varsToSave)

with open(os.path.join(saveDir, 'loggingParameters.pkl'), 'wb') as thisFile:
    pickle.dump(varsToSave, thisFile)

################################################################################
#
####                             GENERAL SETUP
#
################################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 'cuda:0'
else:
    device = 'cpu'

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
l2errorBest = {}
l2errorLast = {}
for thisModel in modelList:
    l2errorBest[thisModel] = [None] * nDataRealizations
    l2errorLast[thisModel] = [None] * nDataRealizations

if doFigs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    costTrain = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nDataRealizations
        costTrain[thisModel] = [None] * nDataRealizations

################################################################################
####                      SPECIFIC TRAINING OPTIONS                         ####
################################################################################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options, then we create a
# separate dictionary per model.

trainingOptsPerModel= {}

################################################################################
#
####                               MAIN RUN
#
################################################################################

# First we create the trajectory, and over the same trajectory we start
# increasing K.

# This means it's the same baseline for all cases

for rlztn in range(nDataRealizations):

    if doPrint:

        print("Running data realization %d..." % rlztn)

################################################################################
####                         For Each Value of sigma2
################################################################################

    # Create the empty places for every model
    for thisModel in modelList:
        l2errorBest[thisModel][rlztn] = [None] * SNRpoints
        l2errorLast[thisModel][rlztn] = [None] * SNRpoints
        lossTrain[thisModel][rlztn] = []
        costTrain[thisModel][rlztn] = []


    for it in range(SNRpoints):

        if doPrint:
            print("[%d] Running for SNR=%ddB..." % (rlztn, SNR[it]))

################################################################################
####                          Create Trajectory
################################################################################

        # Create random matrices
        G = graphTools.Graph('geometric', N, graphOptions) # Create the graph
        G.computeGFT() # Get the eigenvalues for normalization
        A = G.S/np.max(np.real(G.E)) # Matrix A

        Sigmav = particles.createCovarianceMatrix(N, sigma2[it])
        Sigmaw = particles.createCovarianceMatrix(M, sigma2[it])

        xt, yt = particles.createNonlinearTrajectory(T, f, A, C,
                                                     muo, Sigmao,
                                                     muv, Sigmav,
                                                     muw, Sigmaw)

        varsToSave = {'G': G,
                      'A': A,
                      'Sigmav': Sigmav,
                      'Sigmaw': Sigmaw,
                      'xt': xt,
                      'yt': yt}

        with open(os.path.join(dataSplitDir,'dataG%02dSNR%02d.pkl' % (rlztn, SNR[it])), 'wb') as thisFile:
            pickle.dump(varsToSave, thisFile)

        baseline = xt[T-1]

    ########
    #### Learn SIS without resampling
    #####################################

        if doLearnSIS:

            # Initialize the particle filter
            LearnSIS = particles.learnNonlinearSIS(T, f, A, C,
                                                   muo, Sigmao, Sigmav, Sigmaw,
                                                   F, nonlinearity,
                                                   K, yt,
                                                   device = device)

            # Train it
            LearnSIS.baseline = baseline.copy()
            thisLoss, thisCost, thisTime = \
                                LearnSIS.train(learningRate, nEpochs, saveDir,
                                               doPrint = True)

            lossTrain['LearnSIS'][rlztn] += [thisLoss]
            costTrain['LearnSIS'][rlztn] += [thisCost]

            # Save the parameters of LrnRsSIS
            saveModelDir = os.path.join(saveDir,'savedModels')
            # Create directory savedModels if it doesn't exist yet:
            assert os.path.exists(saveModelDir)
            loadFileMeanBest = os.path.join(saveModelDir, 'LearnSIS-Mean' + '-Best-Archit.ckpt')
            loadFileCovBest = os.path.join(saveModelDir, 'LearnSIS-Cov' + '-Best-Archit.ckpt')
            loadFileMeanLast = os.path.join(saveModelDir, 'LearnSIS-Mean' + '-Last-Archit.ckpt')
            loadFileCovLast = os.path.join(saveModelDir, 'LearnSIS-Cov' + '-Last-Archit.ckpt')

    ########
    #### Learn SIS with resampling
    ##################################

        if doLrnRsSIS:

            # Create the new architecture
            LrnRsSIS = particles.learnNonlinearSIS(T, f, A, C,
                                                   muo, Sigmao, Sigmav, Sigmaw,
                                                   F, nonlinearity,
                                                   K, yt, Kthres = Kthres,
                                                   device = device)

        # Create the empty places for every model
        for thisModel in modelList:
            l2errorBest[thisModel][rlztn][it] = [None] * nSampleRealizations
            l2errorLast[thisModel][rlztn][it] = [None] * nSampleRealizations

        for smpl in range(nSampleRealizations):

            if doPrint:
                print("[%d, SNR=%ddB] Running for sample %d..." % (rlztn,
                                                               SNR[it],
                                                               smpl))

            # We will store the estimates here
            estimateBest = {}
            estimateLast = {}

        ########
        #### Learn SIS without resampling
        #####################################

            if doLearnSIS:

                with torch.no_grad():
                    # Best
                    LearnSIS.meanNN.load_state_dict(torch.load(loadFileMeanBest))
                    LearnSIS.covNN.load_state_dict(torch.load(loadFileCovBest))
                    # Create the particles
                    LearnSIS.xt = torch.empty((0, K, N), device = device) # t x K x N
                    LearnSIS.wt = torch.empty((0, K), device = device) # t x K
                    LearnSIS.mut = torch.empty((0, K, N), device = device)
                    LearnSIS.Sigmat = torch.empty((0, K, N, N), device = device)

                    LearnSIS.run()

                    # Compute the estimate
                    #   Get the particles
                    xtLearnSIS = LearnSIS.getParticles() # T x K x N
                    #   Get the weights
                    wtLearnSIS = LearnSIS.getWeights() # T x K
                    #   Multiply the particles by the corresponding weights, add them across
                    #   the K dimension, and get only the last time because we only want an
                    #   estimate of E[x_t|y_{0:t}]
                    estimateBest['LearnSIS'] = np.sum(xtLearnSIS.cpu().numpy() *\
                                                  np.expand_dims(wtLearnSIS.cpu().numpy(), 2),
                                                  axis = 1)[-1]

                    # Last
                    LearnSIS.meanNN.load_state_dict(torch.load(loadFileMeanLast))
                    LearnSIS.covNN.load_state_dict(torch.load(loadFileCovLast))
                    # Create the particles
                    LearnSIS.xt = torch.empty((0, K, N), device = device) # t x K x N
                    LearnSIS.wt = torch.empty((0, K), device = device) # t x K
                    LearnSIS.mut = torch.empty((0, K, N), device = device)
                    LearnSIS.Sigmat = torch.empty((0, K, N, N), device = device)

                    LearnSIS.run()

                    # Compute the estimate
                    #   Get the particles
                    xtLearnSIS = LearnSIS.getParticles() # T x K x N
                    #   Get the weights
                    wtLearnSIS = LearnSIS.getWeights() # T x K
                    #   Multiply the particles by the corresponding weights, add them across
                    #   the K dimension, and get only the last time because we only want an
                    #   estimate of E[x_t|y_{0:t}]
                    estimateLast['LearnSIS'] = np.sum(xtLearnSIS.cpu().numpy() *\
                                                  np.expand_dims(wtLearnSIS.cpu().numpy(), 2),
                                                  axis = 1)[-1]

        ########
        #### Learn SIS with resampling
        ##################################

            if doLrnRsSIS:

                with torch.no_grad():
                    # Best
                    LrnRsSIS.meanNN.load_state_dict(torch.load(loadFileMeanBest))
                    LrnRsSIS.covNN.load_state_dict(torch.load(loadFileCovBest))
                    # Create the particles
                    LrnRsSIS.xt = torch.empty((0, K, N), device = device) # t x K x N
                    LrnRsSIS.wt = torch.empty((0, K), device = device) # t x K
                    LrnRsSIS.mut = torch.empty((0, K, N), device = device)
                    LrnRsSIS.Sigmat = torch.empty((0, K, N, N), device = device)

                    LrnRsSIS.run()

                    # Compute the estimate
                    #   Get the particles
                    xtLrnRsSIS = LrnRsSIS.getParticles() # T x K x N
                    #   Get the weights
                    wtLrnRsSIS = LrnRsSIS.getWeights() # T x K
                    #   Multiply the particles by the corresponding weights, add them across
                    #   the K dimension, and get only the last time because we only want an
                    #   estimate of E[x_t|y_{0:t}]
                    estimateBest['LrnRsSIS'] = np.sum(xtLrnRsSIS.cpu().numpy() *\
                                                  np.expand_dims(wtLrnRsSIS.cpu().numpy(), 2),
                                                  axis = 1)[-1]

                    # Last
                    LrnRsSIS.meanNN.load_state_dict(torch.load(loadFileMeanLast))
                    LrnRsSIS.covNN.load_state_dict(torch.load(loadFileCovLast))
                    # Create the particles
                    LrnRsSIS.xt = torch.empty((0, K, N), device = device) # t x K x N
                    LrnRsSIS.wt = torch.empty((0, K), device = device) # t x K
                    LrnRsSIS.mut = torch.empty((0, K, N), device = device)
                    LrnRsSIS.Sigmat = torch.empty((0, K, N, N), device = device)

                    LrnRsSIS.run()

                    # Compute the estimate
                    #   Get the particles
                    xtLrnRsSIS = LrnRsSIS.getParticles() # T x K x N
                    #   Get the weights
                    wtLrnRsSIS = LrnRsSIS.getWeights() # T x K
                    #   Multiply the particles by the corresponding weights, add them across
                    #   the K dimension, and get only the last time because we only want an
                    #   estimate of E[x_t|y_{0:t}]
                    estimateLast['LrnRsSIS'] = np.sum(xtLrnRsSIS.cpu().numpy() *\
                                                  np.expand_dims(wtLrnRsSIS.cpu().numpy(), 2),
                                                  axis = 1)[-1]

        ########
        #### Optimal SIS without resampling
        #######################################

            if doOptmlSIS:

                # Initialize the particle filter
                OptmlSIS = particles.optimalNonlinearSIS(f, A, C,
                                                         muo, Sigmao,
                                                         Sigmav, Sigmaw,
                                                         K, yt)

                # Create the particles
                OptmlSIS.run()

                # Compute the estimate
                #   Get the particles
                xtOptmlSIS = OptmlSIS.getParticles() # T x K x N
                #   Get the weights
                wtOptmlSIS = OptmlSIS.getWeights() # T x K
                #   Multiply the particles by the corresponding weights, add them across
                #   the K dimension, and get only the last time because we only want an
                #   estimate of E[x_t|y_{0:t}]
                estimateBest['OptmlSIS'] = np.sum(xtOptmlSIS *\
                                              np.expand_dims(wtOptmlSIS, 2),
                                              axis = 1)[-1]
                estimateLast['OptmlSIS'] = estimateBest['OptmlSIS']

        ########
        #### Optimal SIS with resampling
        ####################################

            if doRsmplSIS:

                # Initialize the particle filter
                RsmplSIS = particles.optimalNonlinearSIS(f, A, C,
                                                         muo, Sigmao,
                                                         Sigmav, Sigmaw,
                                                         K, yt,
                                                         Kthres = Kthres)

                # Create the particles
                RsmplSIS.run()

                # Compute the estimate
                #   Get the particles
                xtRsmplSIS = RsmplSIS.getParticles() # T x K x N
                #   Get the weights
                wtRsmplSIS = RsmplSIS.getWeights() # T x K
                #   Multiply the particles by the corresponding weights, add them across
                #   the K dimension, and get only the last time because we only want an
                #   estimate of E[x_t|y_{0:t}]
                estimateBest['RsmplSIS'] = np.sum(xtRsmplSIS *\
                                              np.expand_dims(wtRsmplSIS, 2),
                                              axis = 1)[-1]

                estimateLast['RsmplSIS'] = estimateBest['RsmplSIS']

        ########
        #### Compute the error
        ##########################

            for thisModel in modelList:
                l2errorBest[thisModel][rlztn][it][smpl] = \
                    np.linalg.norm(estimateBest[thisModel] - baseline,ord = 2)/\
                        np.linalg.norm(baseline, ord = 2)

                l2errorLast[thisModel][rlztn][it][smpl] = \
                    np.linalg.norm(estimateLast[thisModel] - baseline,ord = 2)/\
                        np.linalg.norm(baseline, ord = 2)

################################################################################
#
####                               RESULTS
#
################################################################################

#########
#### Save
##############

varsToSave = {'l2errorBest': l2errorBest,
              'l2errorLast': l2errorLast,
              'lossTrain': lossTrain,
              'costTrain': costTrain,
              'sigma2': sigma2,
              'SNR': SNR}

# Save the chosen parameters in the text file
writeVarValues(varsFile, varsToSave)

# Save the corresponding pickle
with open(os.path.join(saveDir, 'results.pkl'), 'wb') as thisFile:
    pickle.dump(varsToSave, thisFile)

# Summarize over both nData and nSamples
meanErrorBest = {}
stdDevErrorBest = {}
meanErrorLast = {}
stdDevErrorLast = {}

for thisModel in modelList:
    l2errorBest[thisModel] = np.array(l2errorBest[thisModel]) # D x K x S
    l2errorLast[thisModel] = np.array(l2errorLast[thisModel]) # D x K x S

    # First, summarize with respect to nSamples
    meanErrorBest[thisModel] = computeStatistic(l2errorBest[thisModel], axis = 2)
    meanErrorLast[thisModel] = computeStatistic(l2errorLast[thisModel], axis = 2)

    # Then with respect to nData
    stdDevErrorBest[thisModel] = np.std(meanErrorBest[thisModel], axis = 0)
    stdDevErrorLast[thisModel] = np.std(meanErrorLast[thisModel], axis = 0)
    meanErrorBest[thisModel] = computeStatistic(meanErrorBest[thisModel], axis = 0)
    meanErrorLast[thisModel] = computeStatistic(meanErrorLast[thisModel], axis = 0)

if len(sigma2) == 1:

    # Print results
    if doPrint:
        print("\nRMSE:")

        for thisModel in modelList:
            print("\t%s: %.4f (+- %.4f) [Best] / %.4f (+- %.4f) [Last]" % (thisModel,
                                                       meanErrorBest[thisModel][0],
                                                       stdDevErrorBest[thisModel][0],
                                                       meanErrorLast[thisModel][0],
                                                       stdDevErrorBest[thisModel][0]))

################################################################################
####                                 Plot
################################################################################

if doFigs:

    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

########
#### Error as a function of K
#################################

    l2errorSNRbest = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for thisModel in modelList:
        plt.errorbar(SNR, meanErrorBest[thisModel],
                     yerr = stdDevErrorBest[thisModel]/fracErrorBar,
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.yscale('log')
    plt.xlabel(r'SNR [dB]', fontsize = fontSize)
    plt.ylabel('$\|\hat{\mathbf{x}} - \mathbf{x}\|_{2}$', fontsize = fontSize)
    plt.legend(modelList, fontsize = legendSize)
    plt.xticks(fontsize = tickSize)
    plt.yticks(fontsize = tickSize)
    l2errorSNRbest.savefig(os.path.join(saveDirFigs,'l2errorSNRbest.pdf'),
                    bbox_inches = 'tight')
    plt.close(fig = l2errorSNRbest)

    l2errorSNRlast = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for thisModel in modelList:
        plt.errorbar(SNR, meanErrorLast[thisModel],
                     yerr = stdDevErrorLast[thisModel]/fracErrorBar,
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.yscale('log')
    plt.xlabel(r'SNR [dB]', fontsize = fontSize)
    plt.ylabel('$\|\hat{\mathbf{x}} - \mathbf{x}\|_{2}$', fontsize = fontSize)
    plt.legend(modelList, fontsize = legendSize)
    plt.xticks(fontsize = tickSize)
    plt.yticks(fontsize = tickSize)
    l2errorSNRlast.savefig(os.path.join(saveDirFigs,'l2errorSNRlast.pdf'),
                    bbox_inches = 'tight')
    plt.close(fig = l2errorSNRlast)

########
#### Training
################

    trainableModels = ['LearnSIS']
    meanLossTrain = {}
    meanCostTrain = {}
    stdDevLossTrain = {}
    stdDevCostTrain = {}

    # Compute the statistics
    for thisModel in trainableModels:
        lossTrain[thisModel] = np.array(lossTrain[thisModel]) # D x K x n
        costTrain[thisModel] = np.array(costTrain[thisModel]) # D x K x n

        # The average is only over the D dimension
        stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis = 0) # K x n
        stdDevCostTrain[thisModel] = np.std(costTrain[thisModel], axis = 0) # K x n
        meanLossTrain[thisModel] = computeStatistic(lossTrain[thisModel], axis = 0)
        meanCostTrain[thisModel] = computeStatistic(costTrain[thisModel], axis = 0)

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs, xAxisMultiplierTrain)

    for it in range(SNRpoints):

        # Downsample axes
        if xAxisMultiplierTrain > 1:
            # Actual selected samples
            selectSamplesTrain = xTrain
            # Go and fetch tem
            for thisModel in trainableModels:
                meanLossTrain[thisModel][it] = meanLossTrain[thisModel][it]\
                                                        [selectSamplesTrain]
                stdDevLossTrain[thisModel][it] = stdDevLossTrain[thisModel][it]\
                                                            [selectSamplesTrain]
                meanCostTrain[thisModel][it] = meanCostTrain[thisModel][it]\
                                                        [selectSamplesTrain]
                stdDevCostTrain[thisModel][it] = stdDevCostTrain[thisModel][it]\
                                                            [selectSamplesTrain]

        # Set axis limits
        if trimAxes:

            maxYaxisLoss = {}
            maxYaxisCost = {}
            multiplierYaxis = 2

            for thisModel in meanLossTrain.keys():

                maxYaxisLoss[thisModel] = 0
                maxYaxisCost[thisModel] = 0

                lastHalfIndexTrain = np.arange(round(len(meanLossTrain[thisModel][it])/2),
                                               len(meanLossTrain[thisModel][it]))

                lastHalfMeanLoss = np.max(np.mean(meanLossTrain[thisModel][it][lastHalfIndexTrain]))

                maxLoss = np.max(meanLossTrain[thisModel][it])

                if maxLoss > multiplierYaxis*lastHalfMeanLoss:
                    maxYaxisLoss[thisModel] = multiplierYaxis*lastHalfMeanLoss

                lastHalfIndexTrain = np.arange(round(len(meanCostTrain[thisModel][it])/2),
                                               len(meanCostTrain[thisModel][it]))

                lastHalfMeanCost = np.max(np.mean(meanCostTrain[thisModel][it][lastHalfIndexTrain]))

                maxCost = np.max(meanCostTrain[thisModel][it])

                if np.max(maxCost) > multiplierYaxis*lastHalfMeanCost:
                    maxYaxisCost[thisModel] = multiplierYaxis*lastHalfMeanCost

            maxYaxisLossMax = np.max(list(maxYaxisLoss.values()))
            maxYaxisCostMax = np.max(list(maxYaxisCost.values()))

        #\\\ LOSS (Training and validation) for EACH MODEL
        for key in meanLossTrain.keys():
            lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
            plt.errorbar(xTrain, meanLossTrain[key][it],
                         yerr = stdDevLossTrain[key][it]/fracErrorBar,
                         color = '#00205B', linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
            plt.xlabel(r'Training steps', fontsize = fontSize)
            plt.ylabel(r'Loss', fontsize = fontSize)
            if trimAxes:
                if maxYaxisLoss[key] > 0:
                    plt.ylim(bottom = 0.8*np.min(meanLossTrain[key][it]),
                             top = maxYaxisLoss[key])
            #plt.legend([r'Training', r'Validation'], fontsize = legendSize)
            plt.title(r'%s' % key, fontsize = fontSize)
            plt.xticks(fontsize = tickSize)
            plt.yticks(fontsize = tickSize)
            lossFig.savefig(os.path.join(saveDirFigs,'loss%s-SNR%03d.pdf' % (key,SNR[it])),
                            bbox_inches = 'tight')
            plt.close(fig = lossFig)

        #\\\ RMSE (Training and validation) for EACH MODEL
        for key in meanCostTrain.keys():
            costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
            plt.errorbar(xTrain, meanCostTrain[key][it],
                         yerr = stdDevCostTrain[key][it]/fracErrorBar,
                         color = '#00205B', linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
            plt.xlabel(r'Training steps', fontsize = fontSize)
            plt.ylabel(r'Error rate', fontsize = fontSize)
            if trimAxes:
                if maxYaxisCost[key] > 0:
                    plt.ylim(bottom = 0.8*np.min(meanCostTrain[key][it]),
                             top = maxYaxisCost[key])
            #plt.legend([r'Training', r'Validation'], fontsize = legendSize)
            plt.title(r'%s' % key, fontsize = fontSize)
            plt.xticks(fontsize = tickSize)
            plt.yticks(fontsize = tickSize)
            costFig.savefig(os.path.join(saveDirFigs,'cost%s-SNR%03d.pdf' % (key,SNR[it])),
                            bbox_inches = 'tight')
            plt.close(fig = costFig)


################################################################################
#
####                           RUNTIME STATISTICS
#
################################################################################

endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

if doPrint:
    print(" ")
    print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                         totalRunTimeM,
                                         totalRunTimeS))

# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("\nSimulation started: %s\n" %
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" %
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))