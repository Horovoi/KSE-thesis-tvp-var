% This function estimates the marginal likelihood of the VAR model in Chan 
% and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [ml, mlstd] = ml_var(Y,store_theta,store_Sig,prior,bigX,M)

disp('Computing marginal likelihood of VAR.... '); 

M = 20*ceil(M/20);
n = size(store_Sig,2);
k = size(store_theta,2);
T = length(Y)/n;

tmp = zeros(n,2);
for i=1:n
    tmp(i,:) = gamfit(1./store_Sig(:,i)); 
end
nuhat = tmp(:,1); Shat = 1./tmp(:,2);
thetahat = mean(store_theta)';
thetastd = chol(cov(store_theta),'lower');
thetapre = cov(store_theta)\speye(k);

big_Sig = zeros(M,n);
for i=1:n
    big_Sig(:,i) = 1./gamrnd(nuhat(i),1./Shat(i),M,1);
end
big_theta = repmat(thetahat',M,1) + (thetastd*randn(k,M))';

store_w = zeros(M,1);

cIS = -.5*k*log(2*pi) - sum(log(diag(thetastd))) + nuhat'*log(Shat) - sum(gammaln(nuhat));
gIS = @(the,s)+ cIS -.5*(the-thetahat)'*thetapre*(the-thetahat) ...
    -(nuhat+1)'*log(s) - sum(Shat./s);   
like_svar = @(the,s) -T*n/2*log(2*pi) - T/2*sum(log(s)) ...
    -.5*(Y-bigX*the)'*((Y-bigX*the)./repmat(s,T,1));

for loop = 1:M
    Sig = big_Sig(loop,:)';    
    theta = big_theta(loop,:)';        
    llike = like_svar(theta,Sig);
    store_w(loop) = llike + prior(theta,Sig) - gIS(theta,Sig);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);

end