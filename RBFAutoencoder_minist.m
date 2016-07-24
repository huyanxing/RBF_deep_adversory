clc;
clear;
close all;

inputSize = 28 * 28;
hiddenSize = 100;  

lambda = 1e-3;
settings.sigmavalue='opt';
settings.sparsityParam=0.05;
settings.beta=0.0002;

trainData = loadMNISTImages('/home/Stroge/Git/DATASETS/mnist/train-images-idx3-ubyte');
batchTrainingSetting.batchNum = 200;
batchTrainingSetting.maxepoch = 4;
options.maxIter = 1000;

OptTheta = RBFAutoencoderTrain_minibatch(inputSize, hiddenSize, lambda, trainData, settings, batchTrainingSetting, options);