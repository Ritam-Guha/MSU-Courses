data = importdata('diabetes.mat');
x_train = data.x_train;
y_train = data.y_train;
x_test = data.x_test;
y_test = data.y_test;
num_folds = 5;
normalization_type = 'none';

if(normalization_type ~= 'none')
    x_train = normalize(x_train, normalization_type);
    y_train = normalize(y_train, normalization_type);
    x_test = normalize(x_test, normalization_type);
    y_test = normalize(y_test, normalization_type);
end


% Calling the Ridge Regression Solver
RidgeReg(x_train, y_train, x_test, y_test, num_folds, normalization_type);

%% Function for solving Ridge Regression
function [] = RidgeReg(x_train, y_train, x_test, y_test, cross_valid_k, normalization_type)
    num_feat = size(x_train, 2);
    lambda_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]; % possible lambda values
    num_lambdas = size(lambda_vals,2);
    
    for i=1:num_lambdas
        [train_error, test_error] = compute_error(x_train, y_train, x_test, y_test, lambda_vals(1,i));
        fprintf('lambda = %f: Train MSE:%f Test MSE:%f\n', lambda_vals(1,i), train_error, test_error);
    end
    fprintf('\n')
    
    lambda = cross_validate(x_train, y_train, lambda_vals, cross_valid_k, normalization_type);
    fprintf('Best Lambda: %f\n', lambda);
    
    phi = x_train;
    t = y_train;
    
    w_reg = (phi' * phi + lambda * eye(num_feat, num_feat))^-1 * (phi' * t); % weights with regularization
    w_noreg = (phi' * phi)^-1 * (phi' * t);    % weights without regularization
    w_reg_matlab = ridge(t, phi, lambda);   % weights for intgreated ridge regression in MATLAB
    
    
    test_phi = x_test;
    test_t = y_test;
            
    predictions_reg = test_phi * w_reg; % predictions for regularized version
    predictions_noreg = test_phi * w_noreg; % predictions for non-regularized version
    predictions_reg_matlab = test_phi * w_reg_matlab;   % predictions for matlab ridge regression
    
    MSE_reg = mean((predictions_reg - test_t).^2);
    MSE_noreg = mean((predictions_noreg - test_t).^2);
    MSE_reg_matlab = mean((predictions_reg_matlab - test_t).^2);
    
    fprintf('Final MSE for non-regularized variant: %f\n', MSE_noreg);
    fprintf('Final MSE for ridge regularized variant: %f\n', MSE_reg);
    fprintf('Final MSE for integrated MATLAB ridge regularized variant: %f\n', MSE_reg_matlab);
end

%% Function for performing cross validation on the training data
function[best_lambda] = cross_validate(x, y, lambda_vals, k, normalization_type) 
    fprintf('Starting Cross-validation....')
    pause(3)    % Just for dramatic effect
    indices = crossvalind('Kfold',y,k); % generating indices for different folds in K-fold cross-validation
    best_lambda = -1;
    best_lambda_MSE = inf;
    num_feat = size(x,2);
    num_lambdas = size(lambda_vals,2);
    lambda_MSE = zeros(1, num_lambdas);
    
    for lambda_idx = 1:length(lambda_vals)
        cur_lambda = lambda_vals(lambda_idx);
        
        fprintf('\n=================================================\n');
        fprintf('Current Lambda Value: %f', cur_lambda);
        fprintf('\n=================================================\n');
        
        cur_lambda_MSE = 0;
        
        for i = 1:k
            % cross-validation begins
            test_indices = (indices == i);  % the test fold is marked with i
            train_indices = ~test_indices;  % all the other folds are used for training
            
            phi = x(train_indices,:);
            t = y(train_indices, :);
            
            test_phi = x(test_indices,:);
            test_t = y(test_indices,:);
            
            w = (phi' * phi + cur_lambda * eye(num_feat, num_feat))^-1 * (phi' * t);    % weights for ridge regression
            
            predictions = test_phi * w;

            cur_MSE = mean((predictions - test_t).^2);  % calculate MSE for current fold
            cur_lambda_MSE = cur_lambda_MSE + cur_MSE;  
            
            fprintf('MSE for fold %d: %f\n', i, cur_MSE);
        end
        
        cur_mean_MSE = cur_lambda_MSE/k;
        lambda_MSE(1, lambda_idx) = cur_mean_MSE;
        
        fprintf('Mean MSE for Lambda=%f: %f', cur_lambda, cur_mean_MSE);
        fprintf('\n=================================================\n');
        
        if(cur_mean_MSE < best_lambda_MSE)
            best_lambda_MSE = cur_mean_MSE;
            best_lambda = cur_lambda;
        end
    end
    
    % Plotting MSE vs Lambda curve
    x=log10(lambda_vals);
    plot(x, lambda_MSE)
    xlabel('log10(\lambda)')
    ylabel('MSE')
    xticklabels(x)
    title({'Cross-validation MSE for different values of \lambda', strcat('Type of Normalization: ', normalization_type)})
    saveas(gcf, strcat('MSE_Plots_',normalization_type,'.jpg'));
end

%% helper function to compute train and test MSE
function[train_MSE, test_MSE] = compute_error(x_train, y_train, x_test, y_test, lambda)
    num_feat = size(x_train,2);

    phi = x_train;
    t = y_train;
    
    test_phi = x_test;
    test_t = y_test;
    
    w = (phi' * phi + lambda * eye(num_feat, num_feat))^-1 * (phi' * t);
    
    train_predictions = phi * w;
    test_predictions = test_phi * w;
    
    train_MSE = mean((train_predictions - t).^2);
    test_MSE = mean((test_predictions - test_t).^2);
end