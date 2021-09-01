function print_beta_weights (beta, names)
neg_idx = find (beta < 0);
beta = beta (neg_idx);
names = names(neg_idx);
[sorted_beta, sort_idx] = sort (beta, 'ascend');
for i = 1:length (neg_idx)
    fprintf ('%g\t%s\n', beta(sort_idx(i)), names{sort_idx(i)});
end