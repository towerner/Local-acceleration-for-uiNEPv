function [v,d] = geteigs(v,d,shift,k)
% Get k desired eigenvalues closest to shift and corresponding eigenvectors
% Shift can be either numeric, in which case we return eigenpairs closest
% to the shift (in magnitude), or one of 
% SR (smallest real), SM (smallest magnitude), LR (largest real), LM 
% (largest magnitude)

% Sort eigenvalues depending on shift
if isnumeric(shift)
    [~,ix] = sort(d-shift,'ascend','ComparisonMethod','abs');
else
    switch lower(shift)
        case 'sr'
            [~,ix] = sort(d,'ascend','ComparisonMethod','real');
        case 'sm'
            [~,ix] = sort(d,'ascend','ComparisonMethod','abs');
        case 'lr'
            [~,ix] = sort(d,'descend','ComparisonMethod','real');
        case 'lm'
            [~,ix] = sort(d,'descend','ComparisonMethod','abs');
        otherwise
            error('Invalid shift')
    end
end
% Extract desired eigenpairs
d = d(ix(1:k));
v = v(:,ix(1:k));
end
