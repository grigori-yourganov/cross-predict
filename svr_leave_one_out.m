function [r, beta_map, predicted_labels, p] = svr_leave_one_out (data, labels, clipping)
% predict the scores using linear support vector regression
% INPUTS:
% data (#observations x #features) -- data matrix (ROI or voxelwise)
% labels -- scores to predict (i.e. values of dependent variable)
% clipping -- 1 if we only want to keep positive feature weights;
%            -1 if we only want to keep negative feature weights; 
%             0 if we want to keep positive and negative feature weights (default)
% OUTPUTS:
% r -- prediction accuracy (correlation between actual and predicted scores)
% z_map -- map of Z-scored feature weights (features being ROIs or voxels)
% predicted_labels -- predicted score values per observation
% p -- p-value of classification accuracy, assuming binomial distribution

optimize_C = true; 
split_half = true; % use split-half resamplinbg to optimize C (recommended; otherwise, 8-fold CV will be used) 

svmdir = [fileparts(which('NiiStat'))  filesep 'libsvm' filesep];
if ~exist(svmdir,'file') 
    error('Unable to find utility scripts folder %s',svmdir);
end
addpath(svmdir); %make sure we can find the utility scripts

n_subj = length (labels);
predicted_labels = zeros(n_subj,1); %pre-allocate memory
map = zeros(size(data)); %pre-allocate memory

cmd = '-t 0 -s 3';
if optimize_C
    C_list = [0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5];
    N = length (labels);
    if split_half
        N_splits = 20;
        for g = 1:N_splits
            shuffle = randperm (N);
            idx1 = shuffle (1:floor(N/2));
            idx2 = shuffle (floor(N/2)+1:N);
            data1 = data (idx1, :); data2 = data (idx2, :);
            scores1 = labels (idx1); scores2 = labels (idx2);
            for C_idx = 1:length(C_list)
                str = sprintf ('''%s -c %g''', cmd, C_list(C_idx));
                [out, subSVM1] = evalc (['svmtrain (scores1, data1, ' str ');']);
                [out, subSVM2] = evalc (['svmtrain (scores2, data2, ' str ');']); 
                [ww1, bb1] = subGetModelWeights (subSVM1, clipping);
                [ww2, bb2] = subGetModelWeights (subSVM2, clipping);
                sub_prediction{C_idx}(idx2) = ww1*data2' + bb1;
                sub_prediction{C_idx}(idx1) = ww2*data1' + bb2;
                sub_map{C_idx}(2*g-1, :) = ww1;
                sub_map{C_idx}(2*g, :) = ww2;                
            end
        end
    else
        % 8-fold cross validation
        s = floor (N/8);
        G = N - 8*s;
        for C_idx = 1:length(C_list)
            sub_prediction{C_idx} = zeros (size (labels));
        end        
        for g = 1:8
            if g <= G
                subtest_idx = g*s+g-s:g*s+g;
            else
                subtest_idx = g*s+G-s+1:g*s+G;
            end
            subtrain_idx = setdiff (1:N, subtest_idx);
            subtrain_data = data (subtrain_idx, :);
            subtrain_scores = labels (subtrain_idx);
            subtest_data = data (subtest_idx, :);
            subtest_scores = labels (subtest_idx);
            
            for C_idx = 1:length(C_list)
                str = sprintf ('''%s -c %g''', cmd, C_list(C_idx));
                [out, subSVM] = evalc (['svmtrain (subtrain_scores, subtrain_data, ' str ');']);
                [ww, bb] = subGetModelWeights (subSVM, clipping);
                sub_prediction{C_idx}(subtest_idx) = ww*subtest_data' + bb;
                sub_map{C_idx}(g, :) = ww;
            end
        end
    end
    
    for C_idx = 1:length(C_list)
        temp = corrcoef (sub_prediction{C_idx}, labels);
        sub_acc(C_idx) = temp (1, 2);
        temp = sub_map{C_idx};
        sub_repr(C_idx) = mean (mean (corrcoef (temp')));
    end
    cost = ((1+sub_acc)/2).^2 + sub_repr.^2;
    [~, opt_idx] = max (cost);
    C = C_list (opt_idx);
    %fprintf ('Optimized value of C: %g\n', C);
else
    C = 0.01;
    %fprintf ('No optimization of C; using default of %g\n', C);
end
cmd = sprintf ('%s -c %g', cmd, C); 

for subj = 1:n_subj
    train_idx = setdiff (1:n_subj, subj);
    train_data = data (train_idx, :);
    train_labels = labels (train_idx);

    [~, SVM] = evalc(sprintf('svmtrain (train_labels, train_data, ''%s'')',cmd)');
    [out, pred_train_labels] = evalc ('svmpredict (train_labels, train_data, SVM);');

    % step 2 of scale correction: estimate scale&offset
    y = train_labels; %regression line: y = a*x + b
    x = pred_train_labels;
    m = length (train_labels);
    c = (m+1)*sum(x.^2) - sum(x)*sum(x);
    a = ((m+1)*sum(x.*y) - sum(x)*sum(y)) / c;
    b = (sum(x.^2)*sum(y) - sum(x)*sum(x.*y)) / c;
    % predict the test labels
    [ww, bb] = subGetModelWeights (SVM, clipping);
    predicted_labels(subj) = ww*data(subj, :)' + bb;
    % step 3 of scale correction: rescale using estimated scale&offset
    predicted_labels(subj) = a*predicted_labels(subj) + b;
    map (subj, :) = ww; % used to be SVM.sv_coef' * SVM.SVs;
end

[r, p] = corrcoef (predicted_labels', labels);
r = r(1,2); %r = correlation coefficient
p = p(1,2) /2; %p = probability, divide by two to make one-tailed 
if (r < 0.0) %one tailed: we predict predicted scores should positively correlate with observed scores
    p = 1.0-p; %http://www.mathworks.com/matlabcentral/newsreader/view_thread/142056
end
% standardize the maps using jaccknife estimates; GY, Jan 2018; updated May 2018
beta_map = mean (map, 1);
t_map = mean (map, 1) ./ (std (map, 1, 1) * sqrt(n_subj-1));
z_map = zeros (size (t_map));
z_map (:) = nan;
z_map (~isnan (t_map)) = spm_t2z (t_map(~isnan (t_map)), length(labels) - 1);
if exist('good_idx','var') %insert NaN for unused features
    z_mapOK = zeros(n_dim,1);
    z_mapOK(:) = nan;
    z_mapOK(good_idx) = z_map;
    z_map = z_mapOK;
end
%plot results
% figure;
% plot (labels, predicted_labels, 'o');
% axis ([min(labels(:)) max(labels(:)) min(labels(:)) max(labels(:))]);
% %set (gca, 'XTick', [0 1 2 3 4]);
% xlabel ('Actual score');
% ylabel ('Predicted score');
% plot_title = beh_name;
% plot_title (plot_title == '_') = '-';
% title (sprintf ('%s', plot_title));
%end nii_stat_svr_core()


function [ww, bb] = subGetModelWeights (model, clipping)
ww = model.sv_coef' * model.SVs; % model weights
bb = -model.rho; % model offset
if (clipping == 1)
    ww (find (ww < 0)) = 0;
elseif (clipping == -1)
    ww (find (ww > 0)) = 0;
end
% end subGetModelWeights
