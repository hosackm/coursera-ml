function J = computeCostMulti(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
H = X * theta; % Hypothesis for all X rows
J = sum((H - y).^2)/ (2*m); % compute cost function

end
