data = importdata('Data/Spam Email Detection/data.xlsx');   % import data
labels = importdata('Data/Spam Email Detection/labels.xlsx');   % import labels
labels(labels==0) = -1; % transforming 0/1 labels to -1/+1
data(:, size(data,2)+1) = ones(1, size(data,1))'; 
 

global cur_train_size;

% initializing traing and test data
train_size = [200, 500, 800, 1000, 1500, 2000];
test_size = 2601;
[num_samples, num_features] = size(data);
test_data = data(num_samples - test_size + 1 : num_samples, :); 
test_labels = labels(num_samples - test_size + 1 : num_samples);
accuracy = zeros(1, size(train_size, 2));

for i = 1: size(train_size, 2)
    cur_train_size = train_size(i);
    
    train_data = data(1 : cur_train_size, :);
    train_labels = labels(1 : cur_train_size);
    
    weights = logRegression(train_data, train_labels);
    accuracy(1, i) = compute_accuracy(test_data, test_labels, weights);
    fprintf('Training Size = %d: %f\n', cur_train_size, accuracy(1,i));
end

% Plot the variation of accuracy with training size
figure;
X = train_size;
Y = accuracy;
fig = plot(X, Y);
xlabel('Training Size');
ylabel('Classification Accuracy (in %)');
title('Variation of Classification Accuracy with Training Size')
saveas(fig, strcat('Results/Logistic Train/Accuracy_Variance.jpg'));


function [accuracy] = compute_accuracy(data, labels, weights)
    global cur_train_size
    
    % getting the predicted labels and computing accuracy
    predicted_labels = sigmoid(data * weights);
    predicted_labels(predicted_labels > 0.5) = 1;
    predicted_labels(predicted_labels <= 0.5) = -1;
    correct_predictions = sum(predicted_labels == labels);
    accuracy = (correct_predictions/size(data,1) * 100);
    
    % create the confusion matrix
    fig = confusionchart(labels, predicted_labels);
    title(strcat('Confusion Chart for Training Size: ', int2str(cur_train_size)));
    saveas(fig, strcat('Results/Logistic Train/Conf_Chart_Train_Size_', int2str(cur_train_size), '.jpg'));
end

function [weights] = logRegression(data, labels, epsilon, maxiter)
    
    % setting the default parameter values
    if nargin < 4
        if ~exist('epsilon')
            epsilon = 1e-6;
        end
        if ~exist('maxiter')
            maxiter=1000;
        end
    end
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%               iterations tp execute (default=1000)
%   train_size = number of samples used for training
%   test_size = number of samples used for testing
%
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"
    
    
    % initializing parameters
    num_features = size(data, 2);
    w = zeros(num_features, 1);
    iter = 1;
    eta = 0.00001;
    prev_error = Inf;
    cur_error = -Inf;
    
    % loop running till convergence
    while(iter <= maxiter && (abs(cur_error - prev_error) >= epsilon))
        % compute train error
        z = -labels .* (data * w);
        error(1, iter) = mean(log(1 + exp(z)));
        
        prev_error = cur_error;
        cur_error = error(1, iter);
        
        % use the gradient of the loss function wrt the training data to
        % update the weight
        dw = (mean(-exp(-z)./(1+exp(-z)) .* (data .* labels)))';
        w = w - (eta * dw);
        iter = iter+1;
    end
    
    iter = iter-1;
    weights = w;
   
    % plot the train error over the iterations
    figure;
    hold on;
    x = linspace(1,iter,iter);
    plot(x, error);
    legend('Train Error');
    hold off;
end

function[val] = sigmoid(input)
    % sigmoid function implementation
    val = 1./(1 + exp(-input));
end
