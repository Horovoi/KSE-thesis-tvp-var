% This function estimates the marginal likelihood of the TVP-R1-SV model in
% Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [ml, mlstd] = ml_tvp_r1_sv(Y,store_beta,store_Siggam,store_Sigh,...
    store_h0,store_gam0,prior,Xtilde,W,M)

disp('Computing marginal likelihood of TVP-R1-SV.... '); 

M = 20*ceil(M/20);
n = size(store_Sigh,2);
kgam = size(store_Siggam,2);
kbeta = size(store_beta,2);

temp = zeros(kgam,2);
for i=1:kgam
    temp(i,:) = gamfit(1./store_Siggam(:,i));    
end
nugamhat = temp(:,1); Sgamhat = 1./temp(:,2);    
temp = zeros(n,2);
for i=1:n
    temp(i,:) = gamfit(1./store_Sigh(:,i));    
end
nuhhat = temp(:,1); Shhat = 1./temp(:,2);  

betahat = mean(store_beta)';
betastd = chol(cov(store_beta),'lower');
betapre = cov(store_beta)\speye(kbeta);
gam0hat = mean(store_gam0)';
gam0std = chol(cov(store_gam0),'lower');
gam0pre = cov(store_gam0)\speye(kgam);
h0hat = mean(store_h0)';
h0std = chol(cov(store_h0),'lower');
h0pre = cov(store_h0)\speye(n);

big_Siggam = zeros(M,kgam);
for i=1:kgam
    big_Siggam(:,i) = 1./gamrnd(nugamhat(i),1./Sgamhat(i),M,1);
end
big_Sigh = zeros(M,n);
for i=1:n
    big_Sigh(:,i) = 1./gamrnd(nuhhat(i),1./Shhat(i),M,1);
end
big_beta = repmat(betahat',M,1) + (betastd*randn(kbeta,M))';
big_gam0 = repmat(gam0hat',M,1) + (gam0std*randn(kgam,M))';
big_h0 = repmat(h0hat',M,1) + (h0std*randn(n,M))';

store_w = zeros(M,1);
cIS = -.5*kbeta*log(2*pi) - sum(log(diag(betastd))) ...
    + nugamhat'*log(Sgamhat) - sum(gammaln(nugamhat)) ...
    + nuhhat'*log(Shhat) - sum(gammaln(nuhhat)) ...
    -.5*(kgam+n)*log(2*pi) - sum(log(diag(gam0std))) - sum(log(diag(h0std)));
gIS = @(b,sg,sh,a0,b0) cIS -.5*(b-betahat)'*betapre*(b-betahat) ...
    -(nugamhat+1)'*log(sg) - sum(Sgamhat./sg) -(nuhhat+1)'*log(sh) - sum(Shhat./sh) ...
    -.5*(a0-gam0hat)'*gam0pre*(a0-gam0hat) -.5*(b0-h0hat)'*h0pre*(b0-h0hat);

for loop = 1:M
    beta = big_beta(loop,:)';
    Siggam = big_Siggam(loop,:)';
    Sigh = big_Sigh(loop,:)';
    gam0 = big_gam0(loop,:)';
    h0 = big_h0(loop,:)';
    
    llike = intlike_tvpsv(Y-Xtilde*beta,Siggam,Sigh,W,h0,gam0);
    store_w(loop) = llike + prior(beta,Siggam,Sigh,gam0,h0) ...
        - gIS(beta,Siggam,Sigh,gam0,h0);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);

end