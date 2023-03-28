% This function implements the auxiliary mixture sampler to draw the
% log-volatility
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function h = SVRW(Ystar,h,sig,h0)

T = length(h);
    % define normal mixture
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);

    % sample S from a 7-point distrete distribution
tmprand = rand(T,1);
q = repmat(pi,T,1).*normpdf(repmat(Ystar,1,7),repmat(h,1,7)+repmat(mi,T,1), repmat(sqrtsigi,T,1));
q = q./repmat(sum(q,2),1,7);
S = 7 - sum(repmat(tmprand,1,7)<cumsum(q,2),2)+1;
    
    % sample h
Hh =  speye(T) - spdiags(ones(T-1,1),-1,T,T);
iSh = spdiags(1/sig*ones(T,1),0,T,T);
dconst = mi(S)'; iOmega = spdiags(1./sigi(S)',0,T,T);
alph = Hh\[h0;sparse(T-1,1)]; 
HiSH_h = Hh'*iSh*Hh;
Kh = HiSH_h + iOmega;
h_hat = Kh\(HiSH_h*alph + iOmega*(Ystar-dconst));
h = h_hat + chol(Kh,'lower')'\randn(T,1);
end
