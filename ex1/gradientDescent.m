function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % compute Hypothesis
    H = X * theta;
    % allocate space for temp
    temp = zeros(length(theta), 1);

    % iterate over the length of temp
    for i = 1:length(temp)
        % calculate each temp value separately
        temp(i) = theta(i) - alpha * (1/m) * ((H - y)' * X(:, i));
    end

    % update theta with temp calculation
    theta = temp;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
