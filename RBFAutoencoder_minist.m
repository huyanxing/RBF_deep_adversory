 clc;
clear;
close all;

inputSize = 28 * 28;
hiddenSize = 100;  

lambda = 0.02;
settings.sigmavalue=0.5;
settings.sparsityParam=0.05;
settings.beta=0.0002;

trainData = loadMNISTImages('/home/Stroge/Git/DATASETS/mnist/train-images-idx3-ubyte');
batchTrainingSetting.batchNum = 200;
batchTrainingSetting.maxepoch = 1000;
batchTrainingSetting.isStochastic = 0; 
options.maxIter = 20; 

OptTheta = RBFAutoencoderTrain_minibatch(inputSize, hiddenSize, lambda, trainData, settings, batchTrainingSetting, options);