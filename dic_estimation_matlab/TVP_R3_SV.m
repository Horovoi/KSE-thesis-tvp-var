% This script estimates the TVP-R3-SV model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

kgam = n*(n-1)/2;
kbeta = n^2*p;

%% prior
beta0 = zeros(kbeta,1); Vbeta = 10*ones(kbeta,1);
gam0 = zeros(kgam,1); Vgam = 10*ones(kgam,1);
ah = zeros(n,1); Vh = 10*ones(n,1);
amu = zeros(n,1); Vmu = 10*ones(n,1);
numu0 = 5*ones(n,1); Smu0 = .1^2*ones(n,1).*(numu0-1); 
nuh0 = 5*ones(n,1); Sh0 = .01*ones(n,1).*(nuh0-1);

cpri = -.5*kbeta*log(2*pi) - .5*sum(log(Vbeta)) ...
    -.5*kgam*log(2*pi) - .5*sum(log(Vgam)) ...
    + numu0'*log(Smu0) - sum(gammaln(numu0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(2*n)*log(2*pi) - .5*sum(log(Vmu)) - .5*sum(log(Vh));
prior = @(b,g,sm,sh,a0,b0) cpri -.5*((b-beta0)./Vbeta)'*(b-beta0) ...
    -.5*((g-gam0)./Vgam)'*(g-gam0) ...
    - (numu0+1)'*log(sm) - sum(Smu0./sm) ... 
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    - .5*((a0-amu)./Vmu)'*(a0-amu) -.5*((b0-ah)./Vh)'*(b0-ah);

%% compute and define a few things
tmpY = [Y0(end-p+1:end,:); shortY];
X1 = zeros(T,n*p); 
for i=1:p
    X1(:,(i-1)*n+1:i*n) = tmpY(p-i+1:end-i,:);
end
Ztilde = SURform2(X1,n);
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
Z = [Ztilde W];
Hmu = speye(T*n) - sparse(n+1:T*n,1:(T-1)*n,ones((T-1)*n,1),T*n,T*n);

%% initialize
store_Sigmu = zeros(nsims,n); 
store_Sigh = zeros(nsims,n);
store_mu = zeros(nsims,T*n); 
store_gam = zeros(nsims,kgam); 
store_beta = zeros(nsims,kbeta);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);
store_mu0 = zeros(nsims,n);

mu0 = mean(shortY)';
mu = repmat(mu0,T,1);
beta = (Ztilde'*Ztilde)\(Ztilde'*(Y-mu)); 
gam = (W'*W)\(W'*(Y-mu-Ztilde*beta));
Sigmu = .1*ones(n,1);
Sigh = .1*ones(n,1);
h0 = log(mean(reshape(Y - mu - Ztilde*beta - W*gam,n,T).^2,2));
h = repmat(h0',T,1);

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting TVP-R3-SV.... ');
start_time = clock;

for isim = 1:nsims + burnin
      
        %% sample mu
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1)); 
    HinvSH_mu = Hmu'*sparse(1:T*n,1:T*n,repmat(1./Sigmu',1,T))*Hmu;    
    alpmu = Hmu\[mu0;sparse((T-1)*n,1)];
    Kmu = HinvSH_mu + invSig;
    muhat = Kmu\(HinvSH_mu*alpmu + invSig*(Y-Ztilde*beta-W*gam));
    mu = muhat + chol(Kmu,'lower')'\randn(T*n,1);
    
        %% sample beta and gam
    ZinvSig = [Ztilde W]'*invSig;
    Kbeta = sparse(1:kbeta+kgam,1:kbeta+kgam,[1./Vbeta;1./Vgam]) + ZinvSig*[Ztilde W];
    beta_hat = Kbeta\([beta0./Vbeta;gam0./Vgam] + ZinvSig*(Y-mu));
    draw = beta_hat + chol(Kbeta,'lower')'\randn(kbeta+kgam,1);
    beta = draw(1:kbeta); 
    gam = draw(kbeta+1:end);
    
        %% sample mu0
    Kmu0 = sparse(1:n,1:n,1./Sigmu + 1./Vmu);
    mu0hat = Kmu0\(mu0./Vmu + mu(1:n)./Sigmu);
    mu0 = mu0hat + chol(Kmu0,'lower')'\randn(n,1);
        
        %% sample h  
    u = Y-mu-Ztilde*beta-W*gam;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + .0001 );
        h(:,i) = SVRW(Ystar,h(:,i),Sigh(i),h0(i));
    end    
    
        %% sample h0
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);
   
        %% sample Sigmu
    e = reshape(mu-[mu0;mu(1:(T-1)*n)],n,T);
    Sigmu = 1./gamrnd(numu0+T/2, 1./(Smu0 + sum(e.^2,2)/2));
    
        %% sample Sigh
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));
    
    if isim>burnin
        isave = isim-burnin;
        store_mu(isave,:) = mu';
        store_beta(isave,:) = beta';
        store_gam(isave,:) = gam';
        store_h(isave,:) = reshape(h',1,T*n); 
        store_Sigmu(isave,:) = Sigmu';
        store_Sigh(isave,:) = Sigh';        
        store_mu0(isave,:) = mu0';
        store_h0(isave,:) = h0';
    end
    
    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... ' ])
    end 
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

muhat = mean(store_mu)';
muCI = quantile(store_mu,[.05 .95])';
beta_hat = mean(store_beta)';
gam_hat = mean(store_gam)';
gamCI = quantile(store_gam,[.05 .95])';

% figure;
% for i=1:n
%     subplot(2,2,i);
%     hold on
%     if dataflag == 1
%         plot(1:T, truemu(i:n:end), '-','LineWidth',2,'Color','black');
%     end
%     plot(1:T, muhat(i:n:end),  '-','LineWidth',2,'Color','blue');
%     plot(1:T, muCI(i:n:end,1), '--','LineWidth',2,'Color','red');
%     plot(1:T, muCI(i:n:end,2), '--','LineWidth',2,'Color','red');
%     title(['\fontsize{12} \mu_{t,' num2str(i) '}']);
%     hold off   
% end
% hhat = mean(store_h)';
% hCI = quantile(store_h,[.05 .95])';
% l = ceil(n/2);
% figure
% for i=1:n
%     subplot(l,2,i);
%     hold on
%     if dataflag == 1
%         plot(1:T, trueh(:,i), '-','LineWidth',2,'Color','black');
%     end
%     plot(1:T, hhat(i:n:end), '-','LineWidth',2,'Color','blue');
%     plot(1:T, hCI(i:n:end,1), '--','LineWidth',2,'Color','red');
%     plot(1:T, hCI(i:n:end,2), '--','LineWidth',2,'Color','red');
%     title(['\fontsize{14} h_{t,' num2str(i) '}']);
%     hold off   
% end