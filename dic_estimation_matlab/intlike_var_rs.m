% This function evaluates the integrated likelihood of the VAR-RS model in 
% Chan and Eisenstat (2018)
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function llike = intlike_var_rs(shortY,bigX,theta,Sig,P)
r = size(theta,2);
[T,n] = size(shortY);
llike = 0;
like = zeros(T,r);
tmpP1 = zeros(T,r);                   % p(s_t|Y_t,\theta,P)
tmpP2 = zeros(T,r); tmpP2(1,:) = 1/3; % p(s_t|Y_{t-1},\theta,P)

    % compute filtering probabilities 
for i=1:r
    mu_i = bigX*theta(:,i);
    Sig_i = Sig(:,i);
    like(:,i) = mvnpdf(shortY,reshape(mu_i,n,T)',Sig_i');
end
for t=1:T
    if t>1
        tmpP2(t,:) = tmpP1(t-1,:)*P;
    end
    tmp  = tmpP2(t,:) .* like(t,:) ;   
    tmpP1(t,:) = tmp;
    tmpP1(t,:) = tmpP1(t,:)/sum(tmpP1(t,:));
    llike = llike + log(sum(tmp));
end
end