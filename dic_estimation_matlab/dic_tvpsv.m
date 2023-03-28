% This function estimates the DIC of the TVP-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic, pD, Dbar] = dic_tvpsv(Y,store_Sigtheta,store_Sigh,store_h0,...
                store_theta0,bigX,simstep)

disp('Computing DIC of TVP-SV.... '); 
nsims = ceil(size(store_Sigtheta,1)/simstep);
store_llike = zeros(nsims,1);

for isim = 1:nsims
    Sigtheta  = store_Sigtheta((isim-1)*simstep+1,:)';
    Sigh = store_Sigh((isim-1)*simstep+1,:)';
    h0 = store_h0((isim-1)*simstep+1,:)';
    theta0 = store_theta0((isim-1)*simstep+1,:)';
    
    llike = intlike_tvpsv(Y,Sigtheta,Sigh,bigX,h0,theta0);   
    store_llike(isim,:) = llike;        
end    
Dbar = mean(store_llike);

Sigthetahat = mean(store_Sigtheta)';
Sighhat = mean(store_Sigh)';
h0hat = mean(store_h0)';
theta0hat = mean(store_theta0)';
llike = intlike_tvpsv(Y,Sigthetahat,Sighhat,bigX,h0hat,theta0hat,500); 

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;           
end