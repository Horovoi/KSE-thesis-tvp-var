% This script estimates the TVP-R1-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

kgam = n*(n-1)/2;
kbeta = n^2*p + n;    

%% prior
beta0 = zeros(kbeta,1); Vbeta = 10*ones(kbeta,1);
agam = zeros(kgam,1); Vgam = 10*ones(kgam,1);
ah = zeros(n,1); Vh = 10*ones(n,1);
nugam0 = 5*ones(kgam,1); Sgam0 = .01^2*ones(kgam,1).*(nugam0-1); 
nuh0 = 5*ones(n,1); Sh0 = .01*ones(n,1).*(nuh0-1);

cpri = -.5*kbeta*log(2*pi) - .5*sum(log(Vbeta)) ...
    + nugam0'*log(Sgam0) - sum(gammaln(nugam0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(kgam+n)*log(2*pi) - .5*sum(log(Vgam)) - .5*sum(log(Vh));
prior = @(b,sg,sh,a0,b0) cpri -.5*((b-beta0)./Vbeta)'*(b-beta0) ...
    - (nugam0+1)'*log(sg) - sum(Sgam0./sg) ... 
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    - .5*((a0-agam)./Vgam)'*(a0-agam) -.5*((b0-ah)./Vh)'*(b0-ah);

%% compute and define a few things
tmpY = [Y0(end-p+1:end,:); shortY];
X1 = zeros(T,n*p); 
for i=1:p
    X1(:,(i-1)*n+1:i*n) = tmpY(p-i+1:end-i,:);
end
Xtilde = SURform2([ones(T,1) X1],n);
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
W = constructX([],X2,n);
Hgam = speye(T*kgam) - sparse(kgam+1:T*kgam,1:(T-1)*kgam,ones((T-1)*kgam,1),T*kgam,T*kgam);

%% initialize
store_Siggam = zeros(nsims,kgam); 
store_Sigh = zeros(nsims,n);
store_beta = zeros(nsims,kbeta); 
store_gam = zeros(nsims,T*kgam); 
store_gam0 = zeros(nsims,kgam);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);

beta = (Xtilde'*Xtilde)\(Xtilde'*Y);
Siggam = .01*ones(kgam,1);
Sigh = .01*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);
gam0 = zeros(kgam,1);


%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting TVP-R1-SV.... ');
start_time = clock;

for isim = 1:nsims + burnin    
      
        %% sample gam    
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1)); 
    invS = sparse(1:T*kgam,1:T*kgam,repmat(1./Siggam',1,T));
    WinvSig = W'*invSig;
    HinvSH = Hgam'*invS*Hgam;
    alpgam = Hgam\[gam0;sparse((T-1)*kgam,1)];   
    Kgam = HinvSH + WinvSig*W;    
    gamhat = Kgam\(HinvSH*alpgam + WinvSig*(Y-Xtilde*beta));
    gam = gamhat + chol(Kgam,'lower')'\randn(T*kgam,1);
    
        %% sample beta    
    XtildeinvSig = Xtilde'*invSig;
    Kbeta = sparse(1:kbeta,1:kbeta,1./Vbeta) + XtildeinvSig*Xtilde;
    betahat = Kbeta\(beta0./Vbeta + XtildeinvSig*(Y-W*gam));
    beta = betahat + chol(Kbeta,'lower')'\randn(kbeta,1);
    
        %% sample gam0
    Kgam0 = sparse(1:kgam,1:kgam,1./Siggam + 1./Vgam);
    gam0hat = Kgam0\(agam./Vgam + gam(1:kgam)./Siggam);
    gam0 = gam0hat + chol(Kgam0,'lower')'\randn(kgam,1);
        
        %% sample h  
    u = Y-Xtilde*beta - W*gam;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + .0001 );
        h(:,i) = SVRW(Ystar,h(:,i),Sigh(i),h0(i));
    end    
    
        %% sample h0
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);
   
        %% sample Siggam
    e = reshape(gam-[gam0;gam(1:(T-1)*kgam)],kgam,T);
    Siggam = 1./gamrnd(nugam0+T/2, 1./(Sgam0 + sum(e.^2,2)/2));
    
        %% sample Sigh
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));
    
    if isim>burnin
        i = isim-burnin;
        store_beta(i,:) =beta';
        store_gam(i,:) = gam';
        store_h(i,:) = reshape(h',1,T*n); 
        store_Siggam(i,:) = Siggam';
        store_Sigh(i,:) = Sigh';
        store_gam0(i,:) = gam0';
        store_h0(i,:) = h0';        
    end
    
    if ( mod( isim, 10000 ) ==0 )
        disp(  [ num2str( isim ) ' loops... ' ] )
    end 
    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );

betahat = mean(store_beta)';
betaCI = quantile(store_beta,[.05 .95])';
gamhat = mean(store_gam)';
gamCI = quantile(store_gam,[.05 .95])';
