data = importdata('Data/Mall_Customers.csv');
data = data.data;
data = data(:,2:5);
num_clusters = [3:10];
SSE = zeros(1, size(num_clusters,2));
for i=1:size(num_clusters,2)
    [cluster_assignments, cluster_centers] = kmeans_cluster(data, num_clusters(i));
    SSE(i) = compute_SSE(data, cluster_assignments);
    show_plot(data, cluster_assignments);
    saveas(gcf, strcat('Images/KMeans/Cluster_Arrangement_',int2str(num_clusters(i)),'.jpg'));
end
fig = figure; plot(num_clusters, SSE); xlabel('K'); ylabel('SSE'); title('Variation of SSE for different values of K');
saveas(gcf, strcat('Images/KMeans/SSE_Convergence.jpg'));


function [cluster_assignments, cluster_centers] = kmeans_cluster(raw_data, k, spectral)
%     Code to perform k-means clustering
%     INPUTS:
%         data     =  (n * m) matrix where n is the number of samples and m
%                     is the number of features
%         k        =  integer number of intended clusters
%         spectral =  boolean value representing spectral k-means if true,
%                     else standard k-means
% 
%     OUTPUTS:
%         cluster_assignments = labels assigned to the each samples in
%                               [1,k]
%         cluster_centers     = final clsuter centers found in the process
%                               of k-means
    
    % Assigning default values
    if nargin < 3
        if ~exist('spectral')
           spectral=false;
        end
    end
    
    if spectral
        % for spectral relaxation, map the data samples to k-dimensional
        % feature space
        [U, ~, ~] = svd(raw_data);
        projection = U(:, 1:k);
        rand_mat = rand(k,k);
        orth_mat = orth(rand_mat);
        data = projection * orth_mat;
    else
        data = raw_data;
    end
     
    [num_samples, ~] = size(data);
    cluster_assignments = zeros(num_samples, 1);
    temp = randperm(num_samples);
    cluster_center_idx = temp(1:k);
    cluster_centers = data(cluster_center_idx, :);
    change = inf;
    count_iter = 0;
    
    while(change ~= 0)
        % change represents the number cluster assignments that got changed
        % in the current iteration
        count_iter = count_iter+1;
        prev_assignments = cluster_assignments;
        
        for cur_idx=1:num_samples
            min_dist = inf;
            min_idx = -1;
            
            % for each sample, find the cluster center which is at min
            % distance
            for cluster_idx = 1:k
                cur_dist = norm(data(cur_idx,:) - cluster_centers(cluster_idx,:));
                if(cur_dist < min_dist)
                    min_dist = cur_dist;
                    min_idx = cluster_idx;
                end
            end
            cluster_assignments(cur_idx,1) = min_idx;
        end
        
        for cluster_idx = 1:k
            % get the mean of each cluster
            cluster_centers(cluster_idx,:) = mean(data(cluster_assignments == cluster_idx,:));
        end

        change = sum(prev_assignments ~= cluster_assignments);
%         fprintf('Number of changes in iter %d: %d\n', count_iter, change);
%         show_plot(raw_data, cluster_assignments);   % plot the clusters
    end
    
    SSE = compute_SSE(raw_data, cluster_assignments);
    fprintf('Final SSE for k=%d: %f\n', k, SSE);
    
end


function [SSE] = compute_SSE(data, cluster_assignments)
% Function to compute Sum of Squared Error
% INPUTS:
%   data = the dataset used for clustering
%   cluster_assignments = labels for each sample in the data
%   cluster_centers = the centers found for each cluster
% 
% OUTPUT:
%   SSE = final sum of squared errors for the cluster config.

    num_clusters = size(unique(cluster_assignments),1);
    SSE = 0;
    for cluster_no = 1:num_clusters
        cluster_center = mean(data(cluster_assignments == cluster_no,:));
        SSE = SSE + norm(data(cluster_assignments==cluster_no,:)-cluster_center)^2;
    end
end

function [] = show_plot(data, labels)
%   Function to plot the cluster config.
%   INPUTS:
%       data = dataset used for clustering
%       labels = the cluster label assigned to each sample
% 
%   OUTPUT:
%       A plot representing the cluster config.
    
    k = size(unique(labels),1);
    [~, num_features] = size(data);
    
   
    if num_features>2
        pcs = pca(data);
        reduced_data = data * pcs(:, 1:2);
    else
        reduced_data = data;
    end
    
    figure;
    hold on;
    gscatter(reduced_data(:,1),reduced_data(:,2),labels);
    title(strcat('KMeans Clustering Arrangement for K=', int2str(k)));
    hold off;
    pause(2);
end


    