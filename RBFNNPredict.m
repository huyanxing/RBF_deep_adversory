function [pred] = RBFNNPredict(RBFNNModel, features_test)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll the parameters from theta
theta = RBFNNModel.optTheta;
visibleSize = RBFNNModel.inputSize ;
hiddenSize = RBFNNModel.hiddenSize;
numClasses = RBFNNModel.numClasses;
sigmavalue = RBFNNModel.settings.sigmavalue;
sparsityParam = RBFNNModel.settings.sparsityParam;
beta = RBFNNModel.settings.beta;
obj = RBFNNModel.settings.beta;

centroids = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize); % RBF centers

W2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*numClasses), numClasses, hiddenSize);


b2 = theta(hiddenSize*(visibleSize+numClasses)+1:hiddenSize*(visibleSize+numClasses)+numClasses);

if isnumeric(sigmavalue)
   sigma = repmat(sigmavalue,[1,hiddenSize]);
else
   sigma = theta(hiddenSize*(visibleSize+numClasses)+numClasses+1:hiddenSize*(visibleSize+numClasses)+numClasses+hiddenSize);
end
sample_num = size(features_test,2);
%% predict
for i = 1:hiddenSize  % calculate the output node by nodeb
    c_vector = centroids(i,:); % get the center of this node
    c_matrix{i} = repmat(c_vector,[sample_num,1]);
    z2_diff{i} =  features_test - c_matrix{i}';
    z2(i,:) = (arrayfun(@(x)(sum(z2_diff{i}(:,x).^2)),1:size(z2_diff{i},2)))/(2*(sigma(i))^2);
    %clear
end
a2 = exp(-z2);

%calculate the output layer
z3 = W2*a2 + repmat(b2,1,sample_num);

if strcmp(obj,'NonLineraLST')||strcmp(obj,'Softmatrix')
    label_pred = sigmoid(z3); 
elseif strcmp(obj,'LineraLST')
    label_pred = z3; 
end
%%


 % this provides a numClasses x inputSize matrix
label_pred = zeros(1, size(features_test, 2));
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

[~,MaxIndex] = max(label_pred,[],1); 

pred = MaxIndex;
% ---------------------------------------------------------------------

end

