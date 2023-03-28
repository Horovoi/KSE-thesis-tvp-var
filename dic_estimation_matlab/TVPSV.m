% This script estimates the TVP-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

m = n*(n-1)/2;      % dimension of the impact matrix
k = n^2*p + n + m;  % dimension of states

k_Q = .1;
k_W = .1;
k_S = .01;
k_theta = 4; 
k_h = 4;

rng(7, 'twister');

%% prior
% zero priors
%atheta = zeros(k,1);
%Vtheta = 10*ones(k,1);
%ah = zeros(n,1);
%Vh = 10*ones(n,1);

% OLS priors (Primiceri priors)
atheta = data_p(:,2);
Vtheta = k_theta*data_p(:,3);
ah = data_p(1:n,1);
Vh = k_h*ones(n,1);

nutheta0 = 5*ones(k,1); 
Stheta0 = k_Q^2*ones(k,1).*(nutheta0-1); 
Stheta0(1:n*p+1:(k-n*(n-1)/2)) = k_W^2*(nutheta0(1:n*p+1:(k-n*(n-1)/2))-1);
nuh0 = 5*ones(n,1);
Sh0 = k_S*ones(n,1).*(nuh0-1);
cpri = nutheta0'*log(Stheta0) - sum(gammaln(nutheta0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(k+n)*log(2*pi) - .5*sum(log(Vtheta)) - .5*sum(log(Vh));
prior = @(sthe,sh,a0,b0) -(nutheta0+1)'*log(sthe) - sum(Stheta0./sthe) ... 
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    + cpri -.5*((a0-atheta)./Vtheta)'*(a0-atheta) -.5*((b0-ah)./Vh)'*(b0-ah);

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
store_Sigh = zeros(nsims,n);
store_theta = zeros(nsims,T*k); 
store_theta0 = zeros(nsims,k);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);
Sigtheta = .01*ones(k,1);
Sigh = .01*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);
theta0 = zeros(k,1);

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting TVP-SV.... ');
start_time = clock;

for isim = 1:nsims+burnin    
  
    %% sample theta
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1)); 
    invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
    XinvSig = bigX'*invSig;
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
        
    %% sample h  
    u = Y-bigX*theta;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + .0001);
        h(:,i) = SVRW(Ystar,h(:,i),Sigh(i),h0(i));
    end    
    
    %% sample h0
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);
   
    %% sample Sigtheta
    e = reshape(theta-[theta0;theta(1:(T-1)*k)],k,T);
    Sigtheta = 1./gamrnd(nutheta0+T/2, 1./(Stheta0 + sum(e.^2,2)/2));
    
    %% sample Sigh
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));
    
    if isim>burnin
        i = isim-burnin;
        store_theta(i,:) = theta';
        store_h(i,:) = reshape(h',1,T*n); 
        store_Sigtheta(i,:) = Sigtheta';
        store_Sigh(i,:) = Sigh';
        store_theta0(i,:) = theta0';
        store_h0(i,:) = h0';        
    end
    
    if ( mod( isim, 10000 ) ==0 )
        disp(  [ num2str( isim ) ' loops... ' ] )
    end 
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
