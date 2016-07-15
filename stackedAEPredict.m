function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% run the model 
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
thetaL1 = [stack{1}.w(:);stack{1}.b(:)];%% ---------- YOUR CODE HERE --------------------------------------
inputSizeL1 = inputSize;
hiddenSizeL1 = netconfig.layersizes{1};
FeaturesL1 = feedForwardAutoencoder(thetaL1,hiddenSizeL1, inputSizeL1,data);

thetaL2 = [stack{2}.w(:);stack{2}.b(:)];
inputSizeL2 =  netconfig.layersizes{1};
hiddenSizeL2 = netconfig.layersizes{2};
FeaturesL2 = feedForwardAutoencoder(thetaL2,hiddenSizeL2, inputSizeL2,FeaturesL1);

softmaxModel= struct;
softmaxModel.optTheta = softmaxTheta;
pred =softmaxPredict(softmaxModel, FeaturesL2);
% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
