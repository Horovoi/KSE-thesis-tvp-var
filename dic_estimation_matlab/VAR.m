% This script estimates the VAR model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

m = n*(n-1)/2;      % dimension of the impact matrix
k = n^2*p + n + m;  % dimension of states
    
%% prior
atheta = zeros(k,1); Vtheta = 10*ones(k,1);
nu0 = 5*ones(n,1); S0 = ones(n,1).*(nu0-1);

cpri = -.5*k*log(2*pi) - .5*sum(log(Vtheta)) + nu0'*log(S0) - sum(gammaln(nu0));
prior = @(the,s) cpri -.5*(the-atheta)'*((the-atheta)./Vtheta) ...
    -(nu0+1)'*log(s) - sum(S0./s);

%% compute and define a few things
tempY = [Y0(end-p+1:end,:); shortY];
X = zeros(T,n*p); 
for i=1:p
    X(:,(i-1)*n+1:i*n) = tempY(p-i+1:end-i,:);
end
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(i:n:end,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
X1 = SURform2([ones(T,1) X],n); 
bigX = [X1 sparse(X2)];


%% initialize
store_Sig = zeros(nsims,n); 
store_theta = zeros(nsims,k);
Sig = ones(n,1);

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting VAR.... ');
start_time = clock;

for isim = 1:nsims + burnin    
  
    %% sample theta
    XinvSig = bigX'*sparse(1:T*n,1:T*n,repmat(1./Sig,T,1)); 
    Ktheta = sparse(1:k,1:k,1./Vtheta) + XinvSig*bigX;
    theta_hat = Ktheta\(atheta./Vtheta + XinvSig*Y);
    theta = theta_hat + chol(Ktheta,'lower')'\randn(k,1);  
     
    %% sample Sig
    e = reshape(Y - bigX*theta,n,T)';
    Sig = 1./gamrnd(nu0+T/2, 1./(S0 + sum(e.^2)'/2));
    
    if isim>burnin
        i = isim-burnin;
        store_theta(i,:) = theta';
        store_Sig(i,:) = Sig';        
    end
    
    if (mod(isim, 10000) == 0)
        disp([num2str(isim) ' loops... '])
    end 
    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );

theta_hat = mean(store_theta)';
thetaCI = quantile(store_theta,[.05 .95])';