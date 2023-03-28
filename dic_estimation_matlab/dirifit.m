% This function finds the MLE of a dirichlet sample
% flag = 0; optimization fails
function [alphat,flag] = dirifit(X)
[N,k] = size(X);
alphat = mean(X)';
logX = log(X);
err = 1;
flag = 1;
while abs(err) > 1e-4 % stopping criteria
    if min(alphat) < 0
        flag = 0;
        break
    else
        tmpS = repmat(psi(sum(alphat)) - psi(alphat)',N,1) + logX;    
        S = sum(tmpS)';
        H = N*psi(1,sum(alphat)) - N*diag(psi(1,alphat));
        err = - H\S;    
        alphat = alphat + err ;
    end
end
end

