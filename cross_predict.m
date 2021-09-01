clear all
addpath (genpath ('C:/Users/gyourga/Documents/code/NiiStat-master/'));
addpath (genpath ('C:/Users/gyourga/Documents/code/spm12'));

cd C:\Users\gyourga\Documents\code\acute_chronic

clipping = 0;

[filename1, pathname1] = uigetfile({'*.xls;*.xlsx', 'Excel file'}, 'Select dataset 1');
cd (pathname1);
[filename2, pathname2] = uigetfile({'*.xls;*.xlsx', 'Excel file'}, 'Select dataset 2');

[num1, txt1, raw1] = xlsread (filename1);
cd (pathname2);
[num2, txt2, raw2] = xlsread (filename2);

if size (num1, 2) ~= size (num2, 2)
    error (sprintf ('The number of features in %s does not match the number of features in %s! Quitting...', filename1, filename2));
end

name_match = strcmp (txt1(1, 2:size(txt1, 2)-1), txt2(1, 2:size(txt2, 2)-1));
if ~isempty (find (name_match ~= 1))
    error (sprintf ('The names of features in %s do not match the names of features in %s! Quitting...', filename1, filename2));
end  

train_data = num1(:, 1:size(num1, 2) - 1);
train_labels = num1(:, size(num1, 2));
train_labels = train_labels / max (train_labels (:));
subj_idx = find (cell2mat (cellfun (@(x) length(strfind (x, 'mat')), txt1(:, 1), 'UniformOutput', false)));
train_names = txt1 (subj_idx, 1);

test_data = num2(:, 1:size(num2, 2) - 1);
test_labels = num2(:, size(num2, 2));
test_labels = test_labels / max (test_labels (:));
subj_idx = find (cell2mat (cellfun (@(x) length(strfind (x, 'mat')), txt2(:, 1), 'UniformOutput', false)));
test_names = txt2 (subj_idx, 1);

[r12, beta12, pred12, p12] = svr_cross_predict (train_data, train_labels, test_data, test_labels, 0);
[r21, beta21, pred21, p21] = svr_cross_predict (test_data, test_labels, train_data, train_labels, 0);
[r11, beta11, pred11, p11] = svr_leave_one_out (train_data, train_labels, 0);
[r22, beta22, pred22, p22] = svr_leave_one_out (test_data, test_labels, 0);

fprintf ('*******\n\nTraining on %s and predicting %s: \nr = %.4f (p = %g)\n', filename1, filename2, r12, p12);
fprintf ('\nBeta weights for predicting %s from %s:\n', filename2, filename1);
print_beta_weights (beta12, txt1(1, 2:size(txt1, 2)-1));
scatter_plot (test_labels, pred12, ['Training on ' filename1 ', predicting ' filename2]);

fprintf ('*******\n\nTraining on %s and predicting %s: \nr = %.4f (p = %g)\n', filename2, filename1, r21, p21);
fprintf ('\nBeta weights for predicting %s from %s:\n', filename1, filename2);
print_beta_weights (beta21, txt1(1, 2:size(txt1, 2)-1));
scatter_plot (train_labels, pred21, ['Training on ' filename2 ', predicting ' filename1]);

fprintf ('*******\n\nPredicting %s using leave-one-out: \nr = %.4f (p = %g)\n', filename1, r11, p11);
fprintf ('\nBeta weights for predicting %s using leave-one-out:\n', filename1);
print_beta_weights (beta11, txt1(1, 2:size(txt1, 2)-1));
scatter_plot (train_labels, pred11, [filename1 ', leave one out']);

fprintf ('*******\n\nPredicting %s using leave-one-out: \nr = %.4f (p = %g)\n', filename2, r22, p22);
fprintf ('\nBeta weights for predicting %s using leave-one-out:\n', filename2);
print_beta_weights (beta22, txt2(1, 2:size(txt1, 2)-1));
scatter_plot (test_labels, pred22, [filename2 ', leave one out']);

%for i = 1:length(test_labels)
%    fprintf ('%s\t%g\t%g\n', test_names{i}, test_labels(i), pred22(i));
%end
