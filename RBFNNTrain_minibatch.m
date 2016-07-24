function [RBFNNModel] = RBFNNTrain_minibatch(visibleSize, hiddenSize, numClasses, lambda, features, labels, settings, batchTrainingSetting, options)
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
    options.maxIter = 100;
end
sigmavalue = settings.sigmavalue;
sparsityParam = settings.sparsityParam;
beta = settings.beta;
numSamples = length(labels); 
batchNum = batchTrainingSetting.batchNum;
batchSize = numSamples/batchNum;


% Initial batch
indices = randperm(numSamples,batchSize);
batchFeatures = features(:, indices);     
batchLabels = labels(indices);

% initialize parameter Theta 
theta = initializeRBFNNParameters(hiddenSize, visibleSize,numClasses,settings);
% Use minFunc to minimize the function
%fprintf('%6s%12s%12s%12s%12s\n','Iter', 'fObj','fResidue','fSparsity','fWeight');
fprintf('%6s%12s%12s%12s%12s\n','Iter', 'fObj','fResidue','fWeight');
warning off;
maxepoch = batchTrainingSetting.maxepoch;
for iteration = 1:maxepoch
    % Reading the paremeters   
    centroids = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize); % RBF centers
    %centroidsgrad = zeros(size(centroids));
 
    W2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*numClasses), numClasses, hiddenSize);
    %W2grad = zeros(size(W2));

    b2 = theta(hiddenSize*(visibleSize+numClasses)+1:hiddenSize*(visibleSize+numClasses)+numClasses);
    %b2grad = zeros(size(b2));

    if isnumeric(sigmavalue)
       sigma = repmat(sigmavalue,[1,hiddenSize]);
    else
       sigma = theta(hiddenSize*(visibleSize+numClasses)+numClasses+1:hiddenSize*(visibleSize+numClasses)+numClasses+hiddenSize);
       %sigmagrad = zeros(size(sigma));
    end
    
    groundTruth = full(sparse(batchLabels, 1:batchSize, 1));
    
    z2 = zeros(hiddenSize,batchSize); 
    for i = 1:hiddenSize  % calculate the output node by nodeb
        c_vector = centroids(i,:); % get the center of this node
        c_matrix{i} = repmat(c_vector,[batchSize,1]);
        z2_diff{i} =  batchFeatures - c_matrix{i}';
        z2(i,:) = (arrayfun(@(x)(sum(z2_diff{i}(:,x).^2)),1:size(z2_diff{i},2)))/(2*(sigma(i))^2);
    end
    a2 = exp(-z2);
    z3 = W2*a2 + repmat(b2,1,batchSize);
    if strcmp(settings.obj,'NonLineraLST')
        a3 = sigmoid(z3); 
    elseif strcmp(settings.obj,'LineraLST')
        a3 = z3; 
    else
        groundNormal = repmat(sum(exp(z3),1),numClasses,1); %normal term as denominator, also to make sure the sum prob =1
        groundProb = exp(z3);
    end
    if ~strcmp(settings.obj,'Softmax')
        error = (0.5/batchSize)*sum(sum(((groundTruth-a3)').^2));
    else
        error =(-1/batchSize)*sum(sum( groundTruth.*log(groundProb./groundNormal)));
    end
    
    fResidue = error;
    fWeight = 0.5*lambda*(sum(sum(W2.^2))+sum(sum(centroids.^2)));
    %rho = (1/batchSize)*sum(a2,2);
    %fSparsity =  beta*(sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho))));
         %error = weightMatrix * featureMatrix - batchPatches;
        %error = sum(error(:) .^ 2) / batchNumPatches; 
   % fprintf('  %4d  %10.4f  %10.4f  %10.4f  %10.4f\n', iteration, fResidue+fSparsity+fWeight, fResidue, fSparsity, fWeight)
<<<<<<< HEAD
fprintf('%6s%12s%12s%12s \n','Iter', 'fObj','fResidue','fWeight');
    % Initial a new batch
    indices = randperm(numSamples,batchNum);
=======
    fprintf('  %4d  %10.4f  %10.4f  %10.4f\n', iteration, fResidue+fWeight, fResidue, fWeight)
    % Initial a new batchs
    indices = randperm(numSamples,batchSize);
>>>>>>> ff70591de37acce81097715f3d3b81f93fe4b1bf
    batchFeatures = features(:, indices);     
    batchLabels = labels(indices);
    
    options.maxIter = 100;
    addpath minFunc/
    options.Method = 'cg'; %
    options.display = 'on';
    
    fprintf(' ... Start training the %dth minibatch... \n', iteration)
    
    [theta, cost] = minFunc( @(p) RBFNNCost(p, ...
                                   visibleSize, hiddenSize,numClasses,lambda, batchFeatures, batchLabels, settings), ...                                   
                                   theta, options);
     fprintf(' ... Finished training the %dth minibatch... \n', iteration)                           
end

                          
                          
                          
% Fold softmaxOptTheta into a nicer format
RBFNNModel.optTheta = theta;
RBFNNModel.inputSize = visibleSize;
RBFNNModel.hiddenSize = hiddenSize;
RBFNNModel.numClasses = numClasses;
RBFNNModel.settings = settings;
end                          
