% This script estimates the TVP-R2-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

kgam = n*(n-1)/2;
kbeta = n^2*p + n;    

%% prior
gam0 = zeros(kgam,1); Vgam = 10*ones(kgam,1);
abeta = zeros(kbeta,1); Vbeta = 10*ones(kbeta,1);
ah = zeros(n,1); Vh = 10*ones(n,1);
nubeta0 = 5*ones(kbeta,1); 
Sbeta0 = .01^2*ones(kbeta,1).*(nubeta0-1); 
Sbeta0(1:n*p+1:end) = .1^2*(nubeta0(1:n*p+1:end)-1); 
nuh0 = 5*ones(n,1); Sh0 = .01*ones(n,1).*(nuh0-1);

cpri = -.5*kgam*log(2*pi) - .5*sum(log(Vgam)) ...
    + nubeta0'*log(Sbeta0) - sum(gammaln(nubeta0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(kbeta+n)*log(2*pi) - .5*sum(log(Vbeta)) - .5*sum(log(Vh));
prior = @(g,sb,sh,a0,b0) cpri -.5*((g-gam0)./Vgam)'*(g-gam0) ...
    - (nubeta0+1)'*log(sb) - sum(Sbeta0./sb) ... 
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    - .5*((a0-abeta)./Vbeta)'*(a0-abeta) -.5*((b0-ah)./Vh)'*(b0-ah);

%% compute and define a few things
tempY = [Y0(end-p+1:end,:); shortY];
X1 = zeros(T,n*p); 
for i=1:p
    X1(:,(i-1)*n+1:i*n) = tempY(p-i+1:end-i,:);
end
Xtilde = SURform([ones(n*T,1) kron(X1,ones(n,1))]);
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
idj = repmat(1:kgam,1,T)';
idi = zeros(T*kgam,1);
count = 0;
for i=2:n
    for j=1:i-1
        idi(count+1:kgam:end) = i:n:T*n;
        count = count + 1;
    end   
end  
W = sparse(idi,idj,reshape(X2',T*kgam,1));
Hbeta = speye(T*kbeta) - sparse(kbeta+1:T*kbeta,1:(T-1)*kbeta,ones((T-1)*kbeta,1),T*kbeta,T*kbeta);

%% initialize
store_Sigbeta = zeros(nsims,kbeta); 
store_Sigh = zeros(nsims,n);
store_beta = zeros(nsims,T*kbeta); 
store_gam = zeros(nsims,kgam); 
store_beta0 = zeros(nsims,kbeta);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);

gam = (W'*W)\(W'*Y);
Sigbeta = .01*ones(kbeta,1);
Sigh = .01*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);
beta0 = zeros(kbeta,1);

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting TVP-R2-SV.... ');
start_time = clock;

for isim = 1:nsims + burnin    
      
        %% sample beta
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1)); 
    invS = sparse(1:T*kbeta,1:T*kbeta,repmat(1./Sigbeta',1,T));
    XinvSig = Xtilde'*invSig;
    HinvSH = Hbeta'*invS*Hbeta;
    alpbeta = Hbeta\[beta0;sparse((T-1)*kbeta,1)];   
    Kbeta = HinvSH + XinvSig*Xtilde;    
    betahat = Kbeta\(HinvSH*alpbeta + XinvSig*(Y-W*gam));
    beta = betahat + chol(Kbeta,'lower')'\randn(T*kbeta,1);
    
        %% sample gam
    WinvSig = W'*invSig;
    Kgam = sparse(1:kgam,1:kgam,1./Vgam) + WinvSig*W;
    gamhat = Kgam\(gam0./Vgam + WinvSig*(Y-Xtilde*beta));
    gam = gamhat + chol(Kgam,'lower')'\randn(kgam,1);
    
        %% sample beta0
    Kbeta0 = sparse(1:kbeta,1:kbeta,1./Sigbeta + 1./Vbeta);
    beta0hat = Kbeta0\(abeta./Vbeta + beta(1:kbeta)./Sigbeta);
    beta0 = beta0hat + chol(Kbeta0,'lower')'\randn(kbeta,1);
        
        %% sample h  
    u = Y-Xtilde*beta-W*gam;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + .0001 );
        h(:,i) = SVRW(Ystar,h(:,i),Sigh(i),h0(i));
    end    
    
        %% sample h0
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);
   
        %% sample Sigbeta
    e = reshape(beta-[beta0;beta(1:(T-1)*kbeta)],kbeta,T);
    Sigbeta = 1./gamrnd(nubeta0+T/2, 1./(Sbeta0 + sum(e.^2,2)/2));
    
        %% sample Sigh
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));
    
    if isim>burnin
        i = isim-burnin;
        store_beta(i,:) =beta';
        store_gam(i,:) = gam';
        store_h(i,:) = reshape(h',1,T*n); 
        store_Sigbeta(i,:) = Sigbeta';
        store_Sigh(i,:) = Sigh';
        store_beta0(i,:) = beta0';
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