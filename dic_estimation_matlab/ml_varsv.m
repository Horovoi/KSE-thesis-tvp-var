% This function estimates the marginal likelihood of the VAR-SV model in 
% Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [ml, mlstd] = ml_varsv(Y,store_theta,store_Sigh,store_h0,prior,bigX,M)

disp('Computing marginal likelihood of VAR-SV.... '); 

M = 20*ceil(M/20);
n = size(store_Sigh,2);
q = size(store_theta,2);

temp = zeros(n,2);
for i=1:n
    temp(i,:) = gamfit(1./store_Sigh(:,i));    
end
nuhhat = temp(:,1); Shhat = 1./temp(:,2);  
thetahat = mean(store_theta)';
thetastd = chol(cov(store_theta),'lower');
thetapre = cov(store_theta)\speye(q);
h0hat = mean(store_h0)';
h0std = chol(cov(store_h0),'lower');
h0pre = cov(store_h0)\speye(n);

big_Sigh = zeros(M,n);
for i=1:n
    big_Sigh(:,i) = 1./gamrnd(nuhhat(i),1./Shhat(i),M,1);
end
big_theta = repmat(thetahat',M,1) + (thetastd*randn(q,M))';
big_h0 = repmat(h0hat',M,1) + (h0std*randn(n,M))';

store_w = zeros(M,1);
cIS = -.5*(q+n)*log(2*pi) - sum(log(diag(thetastd))) - sum(log(diag(h0std))) ...
    + nuhhat'*log(Shhat) - sum(gammaln(nuhhat));
gIS = @(the,sh,b0) cIS -.5*(the-thetahat)'*thetapre*(the-thetahat) ...
    -(nuhhat+1)'*log(sh) - sum(Shhat./sh) -.5*(b0-h0hat)'*h0pre*(b0-h0hat);

for loop = 1:M
    theta = big_theta(loop,:)';
    Sigh = big_Sigh(loop,:)';    
    h0 = big_h0(loop,:)';    
    
    store_w(loop) = intlike_varsv(Y,theta,Sigh,bigX,h0)...
        + prior(theta,Sigh,h0) - gIS(theta,Sigh,h0);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);

end