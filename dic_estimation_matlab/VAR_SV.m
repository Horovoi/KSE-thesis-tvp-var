% This script estimates the VAR-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

m = n*(n-1)/2;      % dimension of the impact matrix
k = n^2*p + n + m;  % dimension of states

%% prior
atheta = zeros(k,1); Vtheta = 10*ones(k,1);
ah = zeros(n,1); Vh = 10*ones(n,1);
nuh0 = 5*ones(n,1); Sh0 = .01*ones(n,1).*(nuh0-1);

cpri = -.5*(n+k)*log(2*pi) -.5*sum(log(Vtheta)) -.5*sum(log(Vh)) ...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0));
prior = @(the,sh,b0) cpri -.5*(the-atheta)'*((the-atheta)./Vtheta) ...
    -(nuh0+1)'*log(sh) - sum(Sh0./sh) -.5*((b0-ah)./Vh)'*(b0-ah);

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
store_Sigh = zeros(nsims + burnin - burnin,n); 
store_theta = zeros(nsims + burnin - burnin,k);
store_h = zeros(nsims + burnin - burnin,T*n);
store_h0 = zeros(nsims + burnin - burnin,n); 
Sigh = .05*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting VAR-SV.... ');
start_time = clock;

for isim = 1:nsims + burnin    
  
    %% sample theta
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1)); 
    XinvSig = bigX'*invSig; 
    Ktheta = sparse(1:k,1:k,1./Vtheta) + XinvSig*bigX;
    thetahat = Ktheta\(atheta./Vtheta + XinvSig*Y);
    theta = thetahat + chol(Ktheta,'lower')'\randn(k,1);       
    
    %% sample h  
    shortu = reshape(Y-bigX*theta,n,T)';    
    for i=1:n
        Ystar = log(shortu(:,i).^2 + .0001 );
        h(:,i) = SVRW(Ystar,h(:,i),Sigh(i),h0(i));
    end    
    
    %% sample h0
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);
    
    %% sample Sigh
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));  
    
    if isim>burnin
        i = isim-burnin;
        store_h(i,:) = reshape(h',1,T*n); 
        store_theta(i,:) = theta';
        store_Sigh(i,:) = Sigh';   
        store_h0(i,:) = h0';
    end
    
    if ( mod( isim, 10000 ) ==0 )
        disp(  [ num2str( isim ) ' loops... ' ] )
    end 
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );

thetahat = mean(store_theta)';
thetaCI = quantile(store_theta,[.05 .95])';
hhat = mean(store_h)';
hCI = quantile(store_h,[.05 .95])';
