% This function evaluates the integrated likelihood of the TVP model in 
% Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function llike = intlike_tvp(Y,Sig,Sigtheta,bigX,theta0) 
n = size(Sig,1);
Tn = length(Y);
T = Tn/n;
k = size(Sigtheta,1);
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);
invSig = sparse(1:T*n,1:T*n,repmat(1./Sig,T,1)); 
invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
XinvSig = bigX'*invSig;
HinvSH = Htheta'*invS*Htheta;
alptheta = Htheta\[theta0;sparse((T-1)*k,1)];   
Ktheta = HinvSH + XinvSig*bigX;
dtheta = XinvSig*Y + HinvSH*alptheta;    
 
llike = -T*n/2*log(2*pi) - T/2*sum(log(Sigtheta)) - T/2*sum(log(Sig)) ...
    - sum(log(diag(chol(Ktheta)))) - .5*(Y'*invSig*Y ...
    + alptheta'*HinvSH*alptheta - dtheta'*(Ktheta\dtheta)); 
end