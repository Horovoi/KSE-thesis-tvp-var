% This function estimates the marginal likelihood of the regime-switching 
% VAR model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [ml,mlstd] = ml_var_rs_r2(Y,store_theta,store_Sig,store_P,prior,bigX,M)

disp('Computing marginal likelihood of VAR-RS-R2.... '); 

M = 20*ceil(M/20);
r = size(store_P,2);
n = size(store_Sig,2);
k = size(store_theta,2)/r;
T = length(Y)/n;

shortY = reshape(Y,n,T)';
tmp = zeros(n,2);
for i=1:n
    tmp(i,:) = gamfit(1./store_Sig(:,i)); 
end
nuhat = tmp(:,1); Shat = 1./tmp(:,2);
thetahat = mean(store_theta)';
thetastd = chol(cov(store_theta),'lower');
thetapre = cov(store_theta)\speye(k*r);
alphat = zeros(r,r);
for i=1:r
    alphat(i,:) = dirifit(squeeze(store_P(:,i,:)))';
end

big_Sig = zeros(M,n);
for i=1:n
    big_Sig(:,i) = 1./gamrnd(nuhat(i),1./Shat(i),M,1);
end
big_theta = repmat(thetahat',M,1) + (thetastd*randn(k*r,M))';
big_P = zeros(M,r,r);
for i=1:r
    big_P(:,i,:) = dirirnd(alphat(i,:)',M);
end

store_w = zeros(M,1);

cIS = -.5*k*r*log(2*pi) - sum(log(diag(thetastd))) + nuhat'*log(Shat) - sum(gammaln(nuhat));
gIS = @(the,s)+ cIS -.5*(the-thetahat)'*thetapre*(the-thetahat) ...
    -(nuhat+1)'*log(s) - sum(Shat./s);

for isim = 1:M
    Sig = big_Sig(isim,:)';    
    theta = big_theta(isim,:)';        
    P = squeeze(big_P(isim,:,:));
    g_IS_P = 0;
    for i=1:r
        g_IS_P = g_IS_P + ldiripdf(P(i,:),alphat(i,:));
    end        
    llike = intlike_var_rs(shortY,bigX,reshape(theta,k,r),repmat(Sig,1,r),P);
    store_w(isim) = llike + prior(theta,Sig,P) - (gIS(theta,Sig) + g_IS_P);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);

end