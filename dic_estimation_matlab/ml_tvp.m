% This function estimates the marginal likelihood of the TVP model in Chan 
% and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [ml, mlstd] = ml_tvp(Y,store_Sig,store_Sigtheta,store_theta0,prior,bigX,M)

disp('Computing marginal likelihood of TVP.... '); 

M = 20*ceil(M/20);
n = size(store_Sig,2);
k = size(store_Sigtheta,2);

tmp = zeros(k,2);
for i=1:k
    tmp(i,:) = gamfit(1./store_Sigtheta(:,i));    
end
nuthetahat = tmp(:,1); Sthetahat = 1./tmp(:,2);    
tmp = zeros(n,2);
for i=1:n
    tmp(i,:) = gamfit(1./store_Sig(:,i));    
end
nuhat = tmp(:,1); Shat = 1./tmp(:,2);

theta0hat = mean(store_theta0)';
theta0std = chol(cov(store_theta0),'lower');
theta0pre = cov(store_theta0)\speye(k);

big_Sigtheta = zeros(M,k);
for i=1:k
    big_Sigtheta(:,i) = 1./gamrnd(nuthetahat(i),1./Sthetahat(i),M,1);
end
big_Sig = zeros(M,n);
for i=1:n
    big_Sig(:,i) = 1./gamrnd(nuhat(i),1./Shat(i),M,1);
end
big_theta0 = repmat(theta0hat',M,1) + (theta0std*randn(k,M))';

store_w = zeros(M,1);
cIS = nuthetahat'*log(Sthetahat) - sum(gammaln(nuthetahat)) ...
    + nuhat'*log(Shat) - sum(gammaln(nuhat)) -.5*k*log(2*pi) - sum(log(diag(theta0std)));
gIS = @(s,sthe,a0) cIS -(nuthetahat+1)'*log(sthe) - sum(Sthetahat./sthe) ... 
    -(nuhat+1)'*log(s) - sum(Shat./s) -.5*(a0-theta0hat)'*theta0pre*(a0-theta0hat);

for loop = 1:M
    Sigtheta = big_Sigtheta(loop,:)';
    Sig = big_Sig(loop,:)';
    theta0 = big_theta0(loop,:)';    
    
    llike = intlike_tvp(Y,Sig,Sigtheta,bigX,theta0);
    store_w(loop) = llike + prior(Sig,Sigtheta,theta0) - gIS(Sig,Sigtheta,theta0);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);

end