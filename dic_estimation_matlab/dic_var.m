% This function estimates the DIC of the VAR model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [dic, pD, Dbar] = dic_var(Y,store_theta,store_Sig,bigX,simstep)

disp('Computing DIC of VAR.... '); 

nsims = ceil(size(store_theta,1)/simstep);
n = size(store_Sig,2);
T = length(Y)/n;
store_llike = zeros(nsims,1);
like_svar = @(the,s) -T*n/2*log(2*pi) - T/2*sum(log(s)) ...
    -.5*(Y-bigX*the)'*((Y-bigX*the)./repmat(s,T,1));

for isim = 1:nsims
    theta  = store_theta((isim-1)*simstep+1,:)';
    Sig = store_Sig((isim-1)*simstep+1,:)';
        
    llike = like_svar(theta,Sig);
    store_llike(isim,:) = llike;        
end    
Dbar = mean(store_llike);

thetahat = mean(store_theta)';
Sighat = mean(store_Sig)';
llike = like_svar(thetahat,Sighat);

dic = -4*Dbar + 2*llike;
pD = -2*Dbar + 2*llike;
end