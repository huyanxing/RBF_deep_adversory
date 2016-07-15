%%        
%           clc;
%           clear;
           data=rand(10,4);
          neuronnum = 3;
%% deal with missing value
          
          
%%
          [samplenum,featurenum]=size(data);
          visibleSize = featurenum;
          hiddenSize  = neuronnum;
          sparsityParam = 0.05; % desired average activation of the hidden units.
          lambda = 0.03;         % weight decay parameter       
          beta = 0;              % weight of sparsity penalty term       
          %epsilon = 0.1;	       % epsilon for ZCA whitening
%%
%         theta=initializeParameters(hiddenSize, visibleSize);
          [theta]=initializeRBFParameters(hiddenSize, visibleSize, 'opt');
          sigmavalue = 'opt';
          %[data,~] = scale(data);
          data = data'; 
          %[data,~] = scale(data);
          %Rbf_autoencoderCost(theta, visibleSize, hiddenSize, sigma ...
                                                            %lambda, sparsityParam, beta, data)
          