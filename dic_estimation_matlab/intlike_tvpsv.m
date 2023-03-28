% This function evaluates the integrated likelihood of the TVP-SV model in 
% Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function [intlike, store_llike] = intlike_tvpsv(Y,Sigtheta,Sigh,bigX,h0,theta0,R)
n = size(Sigh,1);
T = size(Y,1)/n;
k = size(Sigtheta,1);
m = n*(n-1)/2;
% obtain the mode of the marginal density of h (unconditional on theta)
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);
invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
alptheta = Htheta\[theta0;sparse((T-1)*k,1)];
Hh = speye(T*n) - sparse(n+1:T*n,1:(T-1)*n,ones((T-1)*n,1),T*n,T*n);
HinvSH_h = Hh'*sparse(1:T*n,1:T*n,repmat(1./Sigh,T,1))*Hh;
alph = Hh\[h0;sparse((T-1)*n,1)];
e_h = 1; ht = repmat(h0,T,1);
countout = 0;
while e_h > .1 && countout < 100
    % E-step 
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-ht)',T*n,1)); 
    XinvSig = bigX'*invSig;
    HinvSH_theta = Htheta'*invS*Htheta;
    Ktheta = HinvSH_theta + XinvSig*bigX;
    dtheta = XinvSig*Y + HinvSH_theta*alptheta;
    thetahat = Ktheta\dtheta;    
    CKtheta = chol(Ktheta,'lower')';
    zhat = sum((bigX/CKtheta).^2,2) + (Y-bigX*thetahat).^2;   
    
    % M-step
    e_hj = 1; htt = ht; countin = 0;
    while e_hj> .1 && countin < 1000
        einvhttzhat = exp(-htt).*zhat;        
        gQ = -HinvSH_h*(htt-alph) -.5*(1-einvhttzhat);
        HQ = -HinvSH_h -.5*sparse(1:T*n,1:T*n,einvhttzhat);             
        newhtt = htt - HQ\gQ;
        e_hj = max(abs(newhtt-htt));
        htt = newhtt;
        countin = countin + 1;
    end    
    if countin < 1000
        e_h = max(abs(ht-htt));
        ht = htt;
    end
    countout = countout + 1;
end    

if countout == 100
    ht = repmat(h0,T,1);
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-ht)',T*n,1)); 
    XinvSig = bigX'*invSig;
    HinvSH_theta = Htheta'*invS*Htheta;
    Ktheta = HinvSH_theta + XinvSig*bigX;
    dtheta = XinvSig*Y + HinvSH_theta*alptheta;
    thetahat = Ktheta\dtheta;    
    CKtheta = chol(Ktheta,'lower')';
    zhat = sum((bigX/CKtheta).^2,2) + (Y-bigX*thetahat).^2; 
    einvhttzhat = exp(-ht).*zhat;    
    HQ = -HinvSH_h -.5*sparse(1:T*m,1:T*n,einvhttzhat);    
end
Z = XinvSig'*(Ktheta\bigX');
HH = -.5*Z'.*(eye(n*T)-Z);
Kh = -(HQ+HH);
Cg = chol(Kh,'lower');

%% evaluate the importance weights
c_pri = -T*n/2*log(2*pi) -.5*T*sum(log(Sigh));
c_IS = -T*n/2*log(2*pi) + sum(log(diag(Cg)));
pri_den = @(x) c_pri -.5*(x-alph)'*HinvSH_h*(x-alph);
IS_den = @(x) c_IS -.5*(x-ht)'*Kh*(x-ht);
if nargin == 6
    R = 10;
    store_llike = zeros(R,1);
    for i=1:R
        hc = ht + Cg'\randn(T*n,1);
        shorthc = reshape(hc,n,T)';
        store_llike(i) = deny_h(Y,shorthc,Sigtheta,bigX,theta0) ...
            + pri_den(hc) - IS_den(hc);
    end
        % increase simulation size if the variance of the log-likelihood > 1
    var_llike = var(store_llike)/R;    
    if var_llike > 1
        RR = floor(var_llike);
        store_llike = [store_llike; zeros(R*RR,1)];
        for i=R+1:R*(RR+1)
            hc = ht + Cg'\randn(T*n,1);
            shorthc = reshape(hc,n,T)';
            store_llike(i) = deny_h(Y,shorthc,Sigtheta,bigX,theta0) ...
                + pri_den(hc) - IS_den(hc);
        end
    end
elseif nargin == 7
    store_llike = zeros(R,1);
    for i=1:R
        hc = ht + Cg'\randn(T*n,1);
        shorthc = reshape(hc,n,T)';
        store_llike(i) = deny_h(Y,shorthc,Sigtheta,bigX,theta0) ...
            + pri_den(hc) - IS_den(hc);
    end   
end
maxllike = max(store_llike);
intlike = log(mean(exp(store_llike-maxllike))) + maxllike;

end

function llike = deny_h(Y,h,Sigtheta,bigX,theta0) 
[T,n] = size(h);
k = size(Sigtheta,1);
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);
invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1)); 
invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
XinvSig = bigX'*invSig;
HinvSH = Htheta'*invS*Htheta;
alptheta = Htheta\[theta0;sparse((T-1)*k,1)];   
Ktheta = HinvSH + XinvSig*bigX;
dtheta = XinvSig*Y + HinvSH*alptheta;    
 
llike = -T*n/2*log(2*pi) - T/2*sum(log(Sigtheta)) - .5*sum(sum(h)) ...
    - sum(log(diag(chol(Ktheta)))) - .5*(Y'*invSig*Y ...
    + alptheta'*HinvSH*alptheta - dtheta'*(Ktheta\dtheta)); 
end

