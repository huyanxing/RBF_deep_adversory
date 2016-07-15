function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer, for convience
% to caculate the number to parameters in the output layer
% in 
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables usefulsaeSoftmaxOptTheta
%M = size(data, 2);
sample_num=size(data,2);
groundTruth = full(sparse(labels, 1:sample_num, 1));


%% forward
inputHL1 = stack{1}.w*data+repmat(stack{1}.b,1,sample_num);
outputHL1 = sigmoid(inputHL1);
inputHL2 = stack{2}.w*outputHL1+repmat(stack{2}.b,1,sample_num);
outputHL2 = sigmoid(inputHL2);
inputOL= softmaxTheta*outputHL2;

%***************predicted result***********
%lable_prob = sigmoid(softmaxTheta*outputL2);
%***************************************

groundNormal = repmat(sum(exp(softmaxTheta*outputHL2),1),numClasses,1);
%normal term as denominator, also to make sure the sum prob =1
groundProb = exp(softmaxTheta*outputHL2);
% term as numerator
regterm = (lambda/2)*(sum(sum(softmaxTheta.^2)));
% regulaztion term to makesure the function is convex
cost =(-1/sample_num)*sum(sum( groundTruth.*log(groundProb./groundNormal)))+regterm;
%% start backpropagation for grad************

softmaxThetaGrad= (-1/sample_num)*(( groundTruth-groundProb./groundNormal)*outputHL2')+lambda*softmaxTheta;

errortermOL= (-1)*(groundTruth-groundProb./groundNormal);%.*sigmoidInv(inputOL);

errortermHL2 = (softmaxTheta'*errortermOL).*sigmoidInv(inputHL2);

errortermHL1=(stack{2}.w'*errortermHL2).*sigmoidInv(inputHL1);

stackgrad{1}.w = (1/sample_num)*(errortermHL1*data');

stackgrad{2}.w = (1/sample_num)*(errortermHL2*outputHL1');

stackgrad{1}.b = (1/sample_num)*sum(errortermHL1,2);

stackgrad{2}.b = (1/sample_num)*sum(errortermHL2,2);
                                            

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

