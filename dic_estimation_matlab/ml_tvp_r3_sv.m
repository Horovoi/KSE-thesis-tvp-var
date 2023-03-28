% This function estimates the marginal likelihood of the TVP-R3-SV model in
% Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [ml, mlstd] = ml_tvp_r3_sv(Y,store_beta,store_gam,store_Sigmu,store_Sigh,...
    store_h0,store_mu0,prior,Z,M)

disp('Computing marginal likelihood of TVP-R3-SV.... '); 

M = 20*ceil(M/20);
n = size(store_Sigh,2);
T = size(Y,1)/n;
kbeta = size(store_beta,2);
kgam = size(store_gam,2);

tmp = zeros(n,2);
for i=1:n
    tmp(i,:) = gamfit(1./store_Sigmu(:,i));    
end
numuhat = tmp(:,1); Smuhat = 1./tmp(:,2);    
tmp = zeros(n,2);
for i=1:n
    tmp(i,:) = gamfit(1./store_Sigh(:,i));    
end
nuhhat = tmp(:,1); Shhat = 1./tmp(:,2);  
betahat = mean([store_beta store_gam])';
betastd = chol(cov([store_beta store_gam]),'lower');
betapre = cov([store_beta store_gam])\speye(kgam+kbeta);

mu0hat = mean(store_mu0)';
mu0std = chol(cov(store_mu0),'lower');
mu0pre = cov(store_mu0)\speye(n);
h0hat = mean(store_h0)';
h0std = chol(cov(store_h0),'lower');
h0pre = cov(store_h0)\speye(n);

big_Sigmu = zeros(M,n);
for i=1:n
    big_Sigmu(:,i) = 1./gamrnd(numuhat(i),1./Smuhat(i),M,1);
end
big_Sigh = zeros(M,n);
for i=1:n
    big_Sigh(:,i) = 1./gamrnd(nuhhat(i),1./Shhat(i),M,1);
end
big_beta = repmat(betahat',M,1) + (betastd*randn(kbeta+kgam,M))';
big_mu0 = repmat(mu0hat',M,1) + (mu0std*randn(n,M))';
big_h0 = repmat(h0hat',M,1) + (h0std*randn(n,M))';

store_w = zeros(M,1);
cIS = -.5*(kbeta+kgam)*log(2*pi) - sum(log(diag(betastd))) ...    
    + numuhat'*log(Smuhat) - sum(gammaln(numuhat)) ...
    + nuhhat'*log(Shhat) - sum(gammaln(nuhhat)) ...
    -.5*(2*n)*log(2*pi) - sum(log(diag(mu0std))) - sum(log(diag(h0std)));
gIS = @(b,sm,sh,a0,b0) cIS -.5*(b-betahat)'*betapre*(b-betahat) ...    
    -(numuhat+1)'*log(sm) - sum(Smuhat./sm) -(nuhhat+1)'*log(sh) - sum(Shhat./sh) ...
    -.5*(a0-mu0hat)'*mu0pre*(a0-mu0hat) -.5*(b0-h0hat)'*h0pre*(b0-h0hat);

for loop = 1:M
    beta = big_beta(loop,1:kbeta)';
    gam = big_beta(loop,kbeta+1:end)';
    Sigmu = big_Sigmu(loop,:)';
    Sigh = big_Sigh(loop,:)';
    mu0 = big_mu0(loop,:)';
    h0 = big_h0(loop,:)';        
    llike = intlike_tvpsv(Y-Z*[beta;gam],Sigmu,Sigh,speye(T*n),h0,mu0);    
    store_w(loop) = llike + prior(beta,gam,Sigmu,Sigh,mu0,h0) ...
        - gIS([beta;gam],Sigmu,Sigh,mu0,h0);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);

end