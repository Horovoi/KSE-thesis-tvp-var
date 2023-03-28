% This function estimates the DIC of the TVP-R2-SV model in Chan and 
% Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic, pD, Dbar] = dic_tvp_r2_sv(Y,store_gam,store_Sigbeta,store_Sigh,...
    store_h0,store_beta0,Xtilde,W,simstep)
             
disp('Computing DIC of TVP-R2-SV.... '); 

nsims = ceil(size(store_gam,1)/simstep);
store_llike = zeros(nsims,1);

for isim = 1:nsims
    gam  = store_gam((isim-1)*simstep+1,:)';
    Sigbeta = store_Sigbeta((isim-1)*simstep+1,:)';
    Sigh = store_Sigh((isim-1)*simstep+1,:)';
    h0 = store_h0((isim-1)*simstep+1,:)';
    beta0 = store_beta0((isim-1)*simstep+1,:)';

    llike = intlike_tvpsv(Y-W*gam,Sigbeta,Sigh,Xtilde,h0,beta0);
    store_llike(isim,:) = llike;        
end    
Dbar = mean(store_llike);

gamhat = mean(store_gam)';
Sigbetahat = mean(store_Sigbeta)';
Sighhat = mean(store_Sigh)';
h0hat = mean(store_h0)';
beta0hat = mean(store_beta0)';
llike = intlike_tvpsv(Y-W*gamhat,Sigbetahat,Sighhat,Xtilde,h0hat,beta0hat,500);

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;
end
