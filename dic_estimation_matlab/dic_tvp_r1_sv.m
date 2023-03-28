% This function estimates the DIC of the TVP-R1-SV model in Chan and 
% Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic,pD,Dbar] = dic_tvp_r1_sv(Y,store_beta,store_Siggam,store_Sigh,...
    store_h0,store_gam0,Xtilde,W,simstep)
          
disp('Computing DIC of TVP-R1-SV.... '); 

nsims = ceil(size(store_Siggam,1)/simstep);
store_llike = zeros(nsims,1);

for isim = 1:nsims
    beta  = store_beta((isim-1)*simstep+1,:)';
    Siggam = store_Siggam((isim-1)*simstep+1,:)';
    Sigh = store_Sigh((isim-1)*simstep+1,:)';
    h0 = store_h0((isim-1)*simstep+1,:)';
    gam0 = store_gam0((isim-1)*simstep+1,:)';

    llike = intlike_tvpsv(Y-Xtilde*beta,Siggam,Sigh,W,h0,gam0);
    store_llike(isim,:) = llike;        
end    
Dbar = mean(store_llike);

betahat = mean(store_beta)';
Siggamhat = mean(store_Siggam)';
Sighhat = mean(store_Sigh)';
h0hat = mean(store_h0)';
gam0hat = mean(store_gam0)';
llike = intlike_tvpsv(Y-Xtilde*betahat,Siggamhat,Sighhat,W,h0hat,gam0hat,500);

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;
end
