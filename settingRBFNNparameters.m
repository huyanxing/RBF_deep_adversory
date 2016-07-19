%%        
%           clc;
%           clear;
          features=rand(10,4);
          neuronnum = 3;
          labels = [1,2,3,3,1,1,2,2,3,3];
%% deal with missing value
          
          
%%
          [samplenum,featurenum]=size(features);
          visibleSize = featurenum;
          hiddenSize  = neuronnum;
          numClasses = length(unique(labels)); 
          lambda = 0.001;
          settings.sigmavalue = 'opt'; 
          settings.sparsityParam = 0.05; 
          settings.beta = 0.001;
          %settings.obj = 'NonLineraLST';
          %settings.obj = 'LineraLST'; 
          settings.obj = 'Softmax';
          % desired average activation of the hidden units.
          %epsilon = 0.1;	       % epsilon for ZCA whitening
%%
%         theta=initializeParameters(hiddenSize, visibleSize);
          [theta]=initializeRBFNNParameters(hiddenSize, visibleSize, numClasses,settings);
     
          
          %[data,~] = scale(data);
          features = features'; 
          %[data,~] = scale(data);
          %Rbf_autoencoderCost(theta, visibleSize, hiddenSize, sigma ...
                                                            %lambda, sparsityParam, beta, data)
          