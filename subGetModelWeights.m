function [ww, bb] = subGetModelWeights (model, clipping)
ww = model.sv_coef' * model.SVs; % model weights
bb = -model.rho; % model offset
if (clipping == 1)
    ww (find (ww < 0)) = 0;
elseif (clipping == -1)
    ww (find (ww > 0)) = 0;
end
% end subGetModelWeights