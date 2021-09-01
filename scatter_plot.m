function scatter_plot (labels, predicted_labels, plot_name)
%plot results
figure;
plot (labels, predicted_labels, 'o');
axis ([min(labels(:)) max(labels(:)) min(labels(:)) max(labels(:))]);
%set (gca, 'XTick', [0 1 2 3 4]);
xlabel ('Actual score');
ylabel ('Predicted score');
plot_title = plot_name;
plot_title (plot_title == '_') = '-';
title (sprintf ('%s', plot_title));