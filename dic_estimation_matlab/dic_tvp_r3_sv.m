% This function estimates the DIC of the TVP-R1-SV model in Chan and 
% Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic,pD,Dbar] = dic_tvp_r3_sv(Y,store_beta,store_gam,store_Sigmu,...
    store_Sigh,store_mu0,store_h0,Z,simstep)
          
disp('Computing DIC of TVP-R3-SV.... '); 

nsims = ceil(size(store_gam,1)/simstep);
store_llike = zeros(nsims,1);
Tn = length(Y);

for isim = 1:nsims
    beta  = store_beta((isim-1)*simstep+1,:)';
    gam  = store_gam((isim-1)*simstep+1,:)';
    Sigmu = store_Sigmu((isim-1)*simstep+1,:)';
    Sigh = store_Sigh((isim-1)*simstep+1,:)';
    mu0 = store_mu0((isim-1)*simstep+1,:)';
    h0 = store_h0((isim-1)*simstep+1,:)';
    llike = intlike_tvpsv(Y-Z*[beta;gam],Sigmu,Sigh,speye(Tn),h0,mu0);    
    store_llike(isim,:) = llike;        
end    
Dbar = mean(store_llike);

betahat = mean(store_beta(1:simstep:end,:))';
gamhat = mean(store_gam(1:simstep:end,:))';
Sigmuhat = mean(store_Sigmu(1:simstep:end,:))';
Sighhat = mean(store_Sigh(1:simstep:end,:))';
mu0hat = mean(store_mu0(1:simstep:end,:))';
h0hat = mean(store_h0(1:simstep:end,:))';
llike = intlike_tvpsv(Y-Z*[betahat;gamhat],Sigmuhat,Sighhat,speye(Tn),h0hat,mu0hat,500);

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;
end
