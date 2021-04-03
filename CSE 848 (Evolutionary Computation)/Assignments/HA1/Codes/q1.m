% Testing conditions
max_iter = 2000;
max_func_eval = 5000;
num_runs = 1;
n = 5;  % dimension of the objective function f(x)
a = -5.12;  % lb on x
b = 5.12;   % ub on x

% fminsearch options
options = optimset('MaxIter', max_iter, 'MaxFunEvals', max_func_eval, 'PlotFcns', @optimplotfval);

min_fval = Inf; % intiialize min value of f
min_x = Inf;    % value of x corresponding min f

for run_no = 1:num_runs
    % Create random initial point in [a, b]
    x0 = a + (rand(1, n) * (b-a));
    [x, fval] = fminsearch(@f, x0, options);    % use Nelder-Meads search algorithm

    if(fval < min_fval)
        min_fval = fval;
        min_x = x;
    end   
end

disp(['Min f(x): ', num2str(min_fval)]);
disp(['x: ', num2str(min_x)]);


function [func_val] = f(x)
    % Formulation of n-dimensional Rastrigin function
    alpha = 2.0;   % set the value of alpha 
    n = size(x, 2);
    func_val = 10*n + sum(x.^2 - 10*cos(alpha * pi * x)); 
end

