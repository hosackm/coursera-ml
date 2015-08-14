function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.
m = length(y); % number of training examples
H = sigmoid(X * theta);

J = (-y' * log(H) - (1. - y)' * log(1 - H)) / m;
J = J + theta(2:end)' * theta(2:end) * lambda / (2 * m);

grad = ((H - y)' * X) / m;
grad(2:end) = grad(2:end) + (lambda / m) * theta(2:end)';

end
