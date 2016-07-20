function [cost,grad] = RBFNNCost(theta, visibleSize, hiddenSize,numClasses,lambda, features, labels, settings)
%%                                                         
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% visibleSize: the number of input units 
% hiddenSize: the number of hidden units  
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: training sample. 
% subfeaturenum : To show the number of features in each data matrix
% K: parameter for CCA calculation 
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

sigmavalue = settings.sigmavalue; 
sparsityParam = settings.sparsityParam; 
beta = settings.beta;

centroids = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize); % RBF centers
centroidsgrad = zeros(size(centroids));

W2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*numClasses), numClasses, hiddenSize);
W2grad = zeros(size(W2));

b2 = theta(hiddenSize*(visibleSize+numClasses)+1:hiddenSize*(visibleSize+numClasses)+numClasses);
b2grad = zeros(size(b2));

if isnumeric(sigmavalue)
   sigma = repmat(sigmavalue,[1,hiddenSize]);
else
   sigma = theta(hiddenSize*(visibleSize+numClasses)+numClasses+1:hiddenSize*(visibleSize+numClasses)+numClasses+hiddenSize);
   sigmagrad = zeros(size(sigma));
end

sample_num = size(features,2);
groundTruth = full(sparse(labels, 1:sample_num, 1));
cost = 0;
%% --------disp([numgrad grad]); 



%calculate the RBF layer

for i = 1:hiddenSize  % calculate the output node by nodeb
    c_vector = centroids(i,:); % get the center of this node
    c_matrix{i} = repmat(c_vector,[sample_num,1]);
    z2_diff{i} =  features - c_matrix{i}';
    z2(i,:) = (arrayfun(@(x)(sum(z2_diff{i}(:,x).^2)),1:size(z2_diff{i},2)))/(2*(sigma(i))^2);
    %clear
end
a2 = exp(-z2);

%calculate the output layer
z3 = W2*a2 + repmat(b2,1,sample_num);

if strcmp(settings.obj,'NonLineraLST')
    a3 = sigmoid(z3); 
elseif strcmp(settings.obj,'LineraLST')
    a3 = z3; 
else
    groundNormal = repmat(sum(exp(z3),1),numClasses,1); %normal term as denominator, also to make sure the sum prob =1
    groundProb = exp(z3);
end



%% splite the output

if ~strcmp(settings.obj,'Softmax')
    cost_main = (0.5/sample_num)*sum(sum(((groundTruth-a3)').^2));
else
    cost_main =(-1/sample_num)*sum(sum( groundTruth.*log(groundProb./groundNormal)));
end
 
weight_decay = 0.5*(sum(sum(W2.^2))+sum(sum(centroids.^2)));%the weigh dacay
%rho = (1/sample_num)*sum(a2,2);
%Regterm =  sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));%Sparse regularization term
cost_main =cost_main +lambda*weight_decay;%+beta*Regterm;
% regularization


%error=least squre+regularization 

cost=cost_main;
%****** finish adjusting ************************** 


%% *********start backpropagation for grad************
% errorterm in layer3 from data L-2 norm


if strcmp(settings.obj,'NonLineraLST')
    errorterm_3 = -(groundTruth-a3).*sigmoidInv(z3); 
elseif strcmp(settings.obj,'LineraLST')
    errorterm_3 = -(groundTruth-a3);
else
    errorterm_3 = -(groundTruth-groundProb./groundNormal);
end


W2grad = W2grad + errorterm_3*a2';
W2grad = (1/sample_num).*W2grad + lambda*W2;

b2grad = b2grad+sum(errorterm_3,2);
b2grad = (1/sample_num)*b2grad;

%reg_grad =beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
errorterm_2 = W2'*errorterm_3.*a2; %+ repmat(reg_grad,1,sample_num).*a2;

for j = 1: hiddenSize
    errordiff = errorterm_2(j,:)*(z2_diff{j})';
    centroidsgrad_Update(j,:) = errordiff/(sigma(j)^2);
    
    if isnumeric(sigmavalue) == 0
         errordiffsig = sum(errorterm_2(j,:)*(z2_diff{j}.^2)');
         sigmagrad_Update(j,:) = errordiffsig/(sigma(j)^3);
    end
    
end
centroidsgrad = centroidsgrad + centroidsgrad_Update;
centroidsgrad = (1/sample_num).*centroidsgrad + lambda*centroids;

if isnumeric(sigmavalue) == 0
    sigmagrad = sigmagrad + sigmagrad_Update;
    sigmagrad = (1/sample_num).*sigmagrad;
    grad = [centroidsgrad(:) ; W2grad(:) ; b2grad(:);sigmagrad(:)]; 
else
    grad = [centroidsgrad(:) ; W2grad(:) ; b2grad(:)]; 
end

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
   sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end
