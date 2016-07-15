%% Stacked Autoencoder
clc;
clear;
close all;
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 1e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

maxIterL1=1; 
maxIterL2=1; 
maxIterSoftmax=100;
maxIterFinetune = 400;
%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

t1=cputime
%  d. Use set(0,'RecursionLimit',N)Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
visibleSizeL1 = inputSize; 
addpath ./minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS t\n\no optimize our cost
options.maxIter = maxIterL1;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSizeL1, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...  
                                   beta, trainData), ...
                              sae1Theta, options);
%%======================================================================
%% STEP 2: Train the second sparse autoencoder
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        visibleSizeL1, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
visibleSizeL2 = hiddenSizeL1; 
options.Method = 'lbfgs'; % Here, we use L-BFGS t\n\no optimize our cost
options.maxIter = maxIterL2;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSizeL2, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...  
                                   beta, sae1Features), ...
                              sae2Theta, options);
%%======================================================================
t_after2ndlayer=cputime;
save('2layerresults.mat');
%% use ELM
ELMmark=1
if ELMmark

ELMTrainingFeature= feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                       visibleSizeL2 , sae1Features);
ELMTrainingData=[trainLabels';ELMTrainingFeature];
ELMTrainingData=ELMTrainingData';
%prepare ELM training data
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10
testfeatureL1= feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        visibleSizeL1, testData);
testfeatureL2= feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        visibleSizeL2, testfeatureL1);
ELMTestingData=[testLabels';testfeatureL2];
ELMTestingData=ELMTestingData';
[learn_time, test_time, train_accuracy, test_accuracy]=elm(ELMTrainingData,ELMTestingData,1,1000,'sig')

% 200*600000

else
%% STEP 3: Train the softmax classifier

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                       visibleSizeL2 , sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

inputSizeL2= hiddenSizeL2;
options.maxIter = maxIterSoftmax;
softmaxModel = softmaxTrain(inputSizeL2, numClasses, lambda, ...    
                            sae2Features,trainLabels,options);
saeSoftmaxOptTheta=softmaxModel.optTheta(:); 
t_aftersoftmax=cputime
%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];
t_afterfinetune=cputime
%% ---------------------- start Finetune-------------------------
%lambda = 1e-4;
%!!!!!!!!!!!!!!!!!!!!!!!!change lambda
addpath ./minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS t\n\no optimize our cost
options.maxIter = maxIterFinetune ;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[stackedAEOptTheta, cost] = minFunc( @(p)stackedAECost(p, inputSize, hiddenSizeL2, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels),...
                                              stackedAETheta , options);
t3=cputime
%%======================================================================
%% STEP 6: Test 
%  Instructions: You will neesave('2layerresults.mat');d to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);


[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));%% ---------- YOUR CODE HERE --------------------------------------
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
%stacktime=t2-t1
%finetunetime=t3-t2
%totaltime=t3-t1
end
% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
