%%Softmax
clc;
clear;
close all;
%%======================================================================
%% STEP 0: Initialise constants and parameters
%

inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)
hiddenSize = 121;
lambda = 3e-3; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data

images = loadMNISTImages('/home/remy/Codes/DataSets/mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('/home/remy/Codes/DataSets/mnist/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;

%% debug gradient

DEBUG = false; % Set DEBUG to true when debugging.
    if DEBUG
    inputSize = 8;
    inputData = randn(8, 100);
    labels = randi(10, 100, 1);
end
%%======================================================================
if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);
    disp([numGrad grad]); 
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
end

%%======================================================================
%% STEP 4: Learning parameters
settings.sigmavalue = 'opt'; 
settings.sparsityParam = 1e-4; 
settings.beta = 0;
%settings.obj = 'NonLineraLST';
%settings.obj = 'LineraLST'; 
settings.obj = 'Softmax';
options.maxIter = 1;

batchTrainingSetting.batchNum = 200;
batchTrainingSetting.maxepoch = 1;

RBFNNModel = RBFNNTrain_minibatch(inputSize,  hiddenSize, numClasses, lambda, ...
                            inputData, labels, settings, batchTrainingSetting,options);

%%======================================================================
%% STEP 5: Testing
%

images = loadMNISTImages('/home/remy/Codes/DataSets/mnist/t10k-images-idx3-ubyte');
labels = loadMNISTLabels('/home/remy/Codes/DataSets/mnist/t10k-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;
%size(softmaxModel.optTheta)
%size(inputData)

% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = RBFNNPredict(RBFNNModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
% If your values are too low (accuracy less than 0.91), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
 