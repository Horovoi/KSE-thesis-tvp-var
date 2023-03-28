% This function evaluates the log density of the dirichlet distribution

function lden = ldiripdf(y, alpha)
[n,k] = size(y);
if ~(k == length(alpha))
    error('dimensions do not match ');
end
if size(alpha, 1) < size(alpha, 2)
    alpha = alpha';
end

const = gammaln(sum(alpha)) - sum(gammaln(alpha));
lden = const + log(y)*(alpha - 1); 
end