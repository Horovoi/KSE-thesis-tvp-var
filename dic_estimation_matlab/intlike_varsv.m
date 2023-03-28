% This function evaluates the integrated likelihood of the VAR-SV model in 
% Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

function intlike = intlike_varsv(Y,theta,Sigh,bigX,h0,R)
max_loop = 100;
n = size(Sigh,1);
T = size(Y,1)/n;
% obtain the proposal density 
Hh = speye(T*n) - sparse(n+1:T*n,1:(T-1)*n,ones((T-1)*n,1),T*n,T*n);
HinvSH_h = Hh'*sparse(1:T*n,1:T*n,repmat(1./Sigh,T,1))*Hh;
alph = Hh\[h0;sparse((T-1)*n,1)];
s2 = (Y-bigX*theta).^2;
e_h = 1; ht = repmat(h0,T,1);
count = 0;
while e_h> .01 && count < max_loop
    einvhts2 = exp(-ht).*s2;
    gh = -HinvSH_h*(ht-alph) -.5*(1-einvhts2);
    Gh = -HinvSH_h -.5*sparse(1:T*n,1:T*n,einvhts2);             
    newht = ht - Gh\gh;
    e_h = max(abs(newht-ht));
    ht = newht;
    count = count + 1;
end    

if count == max_loop
    ht = repmat(h0,T,1);
    einvhts2 = exp(-ht).*s2;    
    Gh = -HinvSH_h -.5*sparse(1:T*n,1:T*n,einvhts2);    
end
Kh = -Gh;
CKh = chol(Kh,'lower');

%% evaluate the importance weights
c_pri = -T*n/2*log(2*pi) -.5*T*sum(log(Sigh));
c_IS = -T*n/2*log(2*pi) + sum(log(diag(CKh)));
pri_den = @(x) c_pri -.5*(x-alph)'*HinvSH_h*(x-alph);    
IS_den = @(x) c_IS -.5*(x-ht)'*Kh*(x-ht);
e = Y-bigX*theta;
if nargin == 5
    R = 10;
    store_llike = zeros(R,1);
    for i=1:R
        hc = ht + CKh'\randn(T*n,1);
        llike = -T*n/2*log(2*pi) - .5*sum(hc) ...
            - .5*e'*sparse(1:T*n,1:T*n,exp(-hc))*e;
        store_llike(i) = llike + pri_den(hc) - IS_den(hc);
    end
        % increase simulation size if the variance of the log-likelihood > 1
    var_llike = var(store_llike)/R;    
    if var_llike > 1
        RR = floor(var_llike);
        store_llike = [store_llike; zeros(R*RR,1)];
        for i=R+1:R*(RR+1)
            hc = ht + CKh'\randn(T*n,1);
            llike = -T*n/2*log(2*pi) - .5*sum(hc) ...
                - .5*e'*sparse(1:T*n,1:T*n,exp(-hc))*e;
            store_llike(i) = llike + pri_den(hc) - IS_den(hc);
        end
    end
elseif nargin == 6
    store_llike = zeros(R,1);
    for i=1:R
        hc = ht + CKh'\randn(T*n,1);
        llike = -T*n/2*log(2*pi) - .5*sum(hc) ...
            - .5*e'*sparse(1:T*n,1:T*n,exp(-hc))*e;
        store_llike(i) = llike + pri_den(hc) - IS_den(hc);
    end  
end
maxllike = max(store_llike);
intlike = log(mean(exp(store_llike-maxllike))) + maxllike;

end