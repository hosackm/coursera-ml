function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
figure; hold on;

pos = find(y == 1);
neg = find(y == 0);

scatter(X(pos, 1), X(pos, 2), 'black', '+');
scatter(X(neg, 1), X(neg, 2), 'yellow', 'o');
hold off;

end
