% This script estimates the TVP model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

m = n*(n-1)/2;      % dimension of the impact matrix
k = n^2*p + n + m;  % dimension of states    

%% prior
atheta = zeros(k,1); Vtheta = 10*ones(k,1);
nutheta0 = 5*ones(k,1); 
Stheta0 = .01^2*ones(k,1).*(nutheta0-1); 
Stheta0(1:n*p+1:(k-n*(n-1)/2)) = .1^2*(nutheta0(1:n*p+1:(k-n*(n-1)/2))-1);
nu0 = 5*ones(n,1); S0 = ones(n,1).*(nu0-1);

cpri = nutheta0'*log(Stheta0) - sum(gammaln(nutheta0)) ...
    + nu0'*log(S0) - sum(gammaln(nu0)) - .5*k*log(2*pi) - .5*sum(log(Vtheta));
prior = @(s,sthe,a0) cpri -(nutheta0+1)'*log(sthe) - sum(Stheta0./sthe) ... 
    - (nu0+1)'*log(s) - sum(S0./s) -.5*((a0-atheta)./Vtheta)'*(a0-atheta);

%% compute and define a few things
tempY = [Y0(end-p+1:end,:); shortY];
X = zeros(T,n*p); 
for i=1:p
    X(:,(i-1)*n+1:i*n) = tempY(p-i+1:end-i,:);
end
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
bigX = constructX(X,X2,n);
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);

%% initialize
store_Sigtheta = zeros(nsims,k); 
store_Sig = zeros(nsims,n);
store_theta = zeros(nsims,T*k); 
store_theta0 = zeros(nsims,k);
Sigtheta = .01*ones(k,1);
Sig = ones(n,1);
theta0 = zeros(k,1);

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting TVP.... ');
start_time = clock;

for isim = 1:nsims + burnin    
  
    %% sample theta    
    invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
    XinvSig = bigX'*sparse(1:T*n,1:T*n,repmat(1./Sig,T,1));
    HinvSH = Htheta'*invS*Htheta;
    alptheta = Htheta\[theta0;sparse((T-1)*k,1)];   
    Ktheta = HinvSH + XinvSig*bigX;
    dtheta = XinvSig*Y + HinvSH*alptheta;
    thetahat = Ktheta\dtheta;
    theta = thetahat + chol(Ktheta,'lower')'\randn(T*k,1);
    
    %% sample theta0
    Ktheta0 = sparse(1:k,1:k,1./Sigtheta + 1./Vtheta);
    theta0hat = Ktheta0\(atheta./Vtheta + theta(1:k)./Sigtheta);
    theta0 = theta0hat + chol(Ktheta0,'lower')'\randn(k,1);
        
    %% sample Sig
    e = reshape(Y - bigX*theta,n,T)';
    Sig = 1./gamrnd(nu0+T/2, 1./(S0 + sum(e.^2)'/2));     
   
    %% sample Sigtheta
    e = reshape(theta-[theta0;theta(1:(T-1)*k)],k,T);
    Sigtheta = 1./gamrnd(nutheta0+T/2, 1./(Stheta0 + sum(e.^2,2)/2));    
    
    if isim>burnin
        i = isim-burnin;
        store_theta(i,:) = theta';        
        store_Sigtheta(i,:) = Sigtheta';
        store_Sig(i,:) = Sig';
        store_theta0(i,:) = theta0';        
    end
    
    if ( mod( isim, 10000 ) ==0 )
        disp(  [ num2str( isim ) ' loops... ' ] )
    end 
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
