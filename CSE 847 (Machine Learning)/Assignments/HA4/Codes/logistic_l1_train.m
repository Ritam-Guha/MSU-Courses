% import train and test data
data = importdata('Data/Alzheimers/ad_data.mat');
train_data = data.X_train;
train_labels = data.y_train;
test_data = data.X_test;
test_labels = data.y_test;
features = importdata('Data/Alzheimers/feature_name.mat');

% possible values for the regularization parameter
par = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

global cur_par

% perform sparse regularized logistic regression for all possible
% parameters
for i=1:size(par, 2)
    cur_par = par(i);
    [w, c] = logL1Regression(train_data, train_labels, cur_par);
    scores = sigmoid(test_data * w + c);
   
    [~, ~, ~, AUC(1,i)] = perfcurve(test_labels, scores, 1); % get the AUC
    accuracy(1, i) = compute_accuracy(test_data, test_labels, w, c);
    num_nz_weights(1, i) = nnz(w);
%     fprintf('Par= %d: Non-zero weights: %d, Accuracy: %f, AUC: %f\n', cur_par, num_nz_weights(1, i), accuracy(1, i), AUC(1,i));
    fprintf('%d\t %f\t %f\n', num_nz_weights(1, i), accuracy(1, i), AUC(1,i));
end

% Plot the variation of accuracy, AUC and No. of non-zero weights with training size
figure;
hold on;
X = par;
Y1 = normalize(accuracy, 'norm');
Y2 = normalize(AUC, 'norm');
Y3 = normalize(num_nz_weights, 'norm');
fig = plot(X, Y1);
plot(X, Y2)
plot(X, Y3);
legend('Accuracy', 'AUC', 'No. of Non-zero Weights')
xlabel('Regularization Parameter');
ylabel('Normalized Metric Scores');
title({'Variation in Classification Accuracy, AUC and Number of non-zero weights', 'with Regularization Parameter'})
saveas(fig, strcat('Results/L1 Logistic Train/Metric_Variance.jpg'));
hold off


function [accuracy] = compute_accuracy(data, labels, weights, c)
    global cur_par

    % getting the predicted labels and computing accuracy
    predicted_labels = sigmoid(data * weights + c);
    predicted_labels(predicted_labels > 0.5) = 1;
    predicted_labels(predicted_labels <= 0.5) = -1;
    correct_predictions = sum(predicted_labels == labels);
    accuracy = (correct_predictions/size(data,1) * 100);
    
    % create the confusion matrix
    fig = confusionchart(labels, predicted_labels);

    title(strcat('Confusion Chart for Regularization Parameter: ', string(cur_par)));
    saveas(fig, strcat('Results/L1 Logistic Train/Conf_Chart_Reg_Par_', string(cur_par), '.jpg'));
end


function [w, c] = logL1Regression(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations

[w, c] = LogisticR(data, labels, par, opts);
end


function[val] = sigmoid(input)
    % sigmoid function implementation
    val = 1./(1 + exp(-input));
end