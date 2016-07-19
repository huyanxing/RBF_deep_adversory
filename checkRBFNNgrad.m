  %data = rand(6,20);
  %[featurenum,samplenum] = size(data);
  %hiddenSize = 5;
  %visibleSize = featurenum;
  %theta = initializeParameters(hiddenSize, visibleSize);
  %sparsityParam = 0.05; 
  %desired average activation of the hidden units.
  %lambda = 3e-3;         
  %weight decay parameter       
  %beta = 5;              
  %weight of sparsity penalty term       
  %epsilon = 0.1;
  %K=1;
  %subFeatureNum = [2,2,2];
  
% [cost,grad] = SplitSparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data,subFeatureNum,K);
% Instructions:

%%
clc;
clear;
settingRBFNNparameters;
 num_minibatch =1;

[cost,grad] = RBFNNCost(theta, visibleSize, hiddenSize,numClasses,lambda, features, labels, settings, num_minibatch);

numGrad = computeNumericalGradient( @(x)RBFNNCost(x, visibleSize, hiddenSize,numClasses,lambda, features, labels, settings, num_minibatch), theta);
%%

%%
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff);
    if diff < 1e-8,
        disp ('OK')
    else
        disp ('Difference too large. Check your gradient computation again')
    end