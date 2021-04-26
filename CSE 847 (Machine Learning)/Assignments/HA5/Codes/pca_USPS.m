USPS_data = importdata('Data/USPS.mat');
USPS_mat = USPS_data.A;
p_comb = [10, 50, 100, 200];    % number of principal components considered
num_p = size(p_comb,2);
entry = 1;

% code for simulating PCA reconstruction as many times as user wants
while (entry == 1)
    img_idx = randi(size(USPS_mat,1));
    img_orig = reshape(USPS_mat(img_idx,:), 16, 16);
  
    figure;
    hold on;
    subplot(1,num_p+1,1); imshow(img_orig'); title('Original Image'); 
    for i=1:num_p
        recov_data = pca_svd(USPS_mat, p_comb(1,i));
        img_recov = reshape(recov_data(img_idx,:), 16, 16);
        subplot(1,num_p+1,i+1); imshow(img_recov'); title(strcat('p=', int2str(p_comb(1,i)))); 
    end
    hold off;
    saveas(gcf, strcat('Images/PCA_USPS/plot_',int2str(img_idx),'.jpg'));
    pause(3);
    close;
    
    entry = input('Press 1 for seeing more images, Press 0 to exit: ');
end


function [recov_data] = pca_svd(raw_data, p)
% Function to perform PCA using SVD
%   INPUTS:
%       raw_data = the data used for PCA
%       p = number of principal components to consider
%   
%   OUTPUT:
%       recov_data = data recovered from using p principal componenents

    data = raw_data;
    data = data - mean(data);
    [~,~,PC] = svd(data);
    data = data * PC(:,1:p);
    
    recov_data = data * PC(:, 1:p)';    % reconstructed image
    reconst_error = norm(raw_data - recov_data, 'fro');
    fprintf('The reconstruction error for p=%d is: %f\n', p ,reconst_error);
end
