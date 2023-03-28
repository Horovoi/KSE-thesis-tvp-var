% This function estimates the DIC of the TVP model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic, pD, Dbar] = dic_tvp(Y,store_Sig,store_Sigtheta,...
    store_theta0,bigX,simstep)

disp('Computing DIC of TVP.... '); 
nsims = ceil(size(store_Sigtheta,1)/simstep);
store_llike = zeros(nsims,1);

for isim = 1:nsims
    Sig = store_Sig((isim-1)*simstep+1,:)';
    Sigtheta  = store_Sigtheta((isim-1)*simstep+1,:)';    
    theta0 = store_theta0((isim-1)*simstep+1,:)';    
    llike = intlike_tvp(Y,Sig,Sigtheta,bigX,theta0);
    store_llike(isim,:) = llike;        
end
Dbar = mean(store_llike);

Sigat = mean(store_Sig)';
Sigthetahat = mean(store_Sigtheta)';
theta0hat = mean(store_theta0)';
llike = intlike_tvp(Y,Sigat,Sigthetahat,bigX,theta0hat);

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;
end