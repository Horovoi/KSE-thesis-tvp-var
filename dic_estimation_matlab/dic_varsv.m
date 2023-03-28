% This function estimates the DIC of the VAR-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic, pD, Dbar] = dic_varsv(Y,store_theta,store_Sigh,store_h0,...
    bigX,simstep)

disp('Computing DIC of VAR-SV.... '); 

nsims = ceil(size(store_theta,1)/simstep);
store_llike = zeros(nsims,1);

for isim = 1:nsims
    theta  = store_theta((isim-1)*simstep+1,:)';
    Sigh = store_Sigh((isim-1)*simstep+1,:)';
    h0 = store_h0((isim-1)*simstep+1,:)';
    
    llike = intlike_varsv(Y,theta,Sigh,bigX,h0);
    store_llike(isim,:) = llike;        
end    
Dbar = mean(store_llike);

thetahat = mean(store_theta)';
Sighhat = mean(store_Sigh)';
h0hat = mean(store_h0)';
llike = intlike_varsv(Y,thetahat,Sighhat,bigX,h0hat,500);

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;
end

