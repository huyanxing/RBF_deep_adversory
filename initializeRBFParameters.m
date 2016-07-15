function theta = initializeRBFParameters(hiddenSize, visibleSize,sigmasetting)
%function [theta,centroids] = initializeParameters(hiddenSize, visibleSize)
%% Initialize parameters randomly based on layer sizes.
if nargin<=2
    r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
    centroids = rand(hiddenSize, visibleSize) * 2 * r - r;
    W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
    b2 = zeros(visibleSize, 1);
    theta = [centroids(:) ; W2(:); b2(:)];
else
    r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
    centroids = rand(hiddenSize, visibleSize) * 2 * r - r;
    W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
    b2 = zeros(visibleSize, 1);   
    sigma = rand(1, hiddenSize)* 2 * r - r;
    theta = [centroids(:) ; W2(:); b2(:); sigma(:)];
end
%
end

