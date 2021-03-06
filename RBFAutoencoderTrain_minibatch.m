function [OptTheta] = RBFAutoencoderTrain_minibatch(visibleSize, hiddenSize, lambda, Data, settings, batchTrainingSetting, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 1000;
end

sigmavalue = settings.sigmavalue;
sparsityParam = settings.sparsityParam;
beta = settings.beta;
numSamples = size(Data,2);
if batchTrainingSetting.isStochastic ~= 1
    batchNum = batchTrainingSetting.batchNum;
    batchSize = numSamples/batchNum;
else 
    batchSize = 1;
end


% Initial batch
if batchTrainingSetting.isStochastic ~= 1
    indices = randperm(numSamples,batchSize);
    batchData = Data(:, indices);    
else 
    indices = 1;
    batchData = Data(:, rem(indices,numSamples));    
end
 


% initialize parameter Theta 
if exist('OptTheta')
    theta = OptTheta;
else
    theta = initializeRBFAutoencodeParameters(hiddenSize, visibleSize,settings);
end
% Use minFunc to minimize the function
%fprintf('%6s%12s%12s%12s%12s\n','Iter', 'fObj','fResidue','fSparsity','fWeight');
fprintf('%6s%12s%12s%12s \n','Iter', 'fObj','fResidue','fWeight');
warning off;
maxepoch = batchTrainingSetting.maxepoch;


for iteration = 1:maxepoch
    % Reading the paremeters   
    centroids = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize); % RBF centers
    W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b2 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+visibleSize);

    if isnumeric(sigmavalue)
       sigma = repmat(sigmavalue,[1,hiddenSize]);
    else
       sigma = theta(2*hiddenSize*visibleSize+visibleSize+1:2*hiddenSize*visibleSize+visibleSize+hiddenSize);
    end
    
    z2 = zeros(hiddenSize,batchSize); 
    for i = 1:hiddenSize  % calculate the output node by nodeb
        c_vector = centroids(i,:); % get the center of this node
        c_matrix{i} = repmat(c_vector,[batchSize,1]);
        z2_diff{i} =  batchData - c_matrix{i}';
        z2(i,:) = (arrayfun(@(x)(sum(z2_diff{i}(:,x).^2)),1:size(z2_diff{i},2)))/(2*(sigma(i))^2);
    end
    a2 = exp(-z2);
    z3 = W2*a2 + repmat(b2,1,batchSize);
    a3 = sigmoid(z3); 
    error = (0.5/batchSize)*sum(sum(((batchData-a3)').^2));

    
    fResidue = error;
    fWeight = 0.5*lambda*(sum(sum(W2.^2))+sum(sum(centroids.^2)));
    %rho = (1/batchSize)*sum(a2,2);
    %fSparsity =  beta*(sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho))));
    %fSparsity = 0; 
    %fprintf('  %4d  %10.4f  %10.4f  %10.4f  %10.4f\n', iteration, fResidue+fSparsity+fWeight, fResidue, fSparsity, fWeight) 
    fprintf('  %4d  %10.4f  %10.4f  %10.4f\n', iteration, fResidue+fWeight, fResidue, fWeight)
    % Initial a new batch
    
    if batchTrainingSetting.isStochastic ~= 1
        indices = randperm(numSamples,batchSize);
        batchData = Data(:, indices);  
    elseif  rem(indices,numSamples) ~= 0
        batchData = Data(:, rem(indices,numSamples)); 
    else
        batchData = Data(:, end);      
    end
    
%    options.maxIter = 20;
    addpath minFunc/
    options.Method = 'lbfgs'; %
    options.display = 'off';
    
    fprintf(' ... Start training the %dth minibatch... \n', iteration)
    
    [theta, cost] = minFunc( @(p) RBFAutoencoderCost(p, ...
                                   visibleSize, hiddenSize,lambda,settings,  batchData), ...                                   
                                   theta, options);
    if batchTrainingSetting. isStochastic == 1                           
    indices = indices+1;
    end
    fprintf(' ... Finished training the %dth minibatch... \n', iteration)                           
end

         
                          
                          
% Fold softmaxOptTheta into a nicer format
OptTheta = theta;
end                          
