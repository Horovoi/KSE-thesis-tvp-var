% This script estimates the RS-VAR-R2 model in Chan and Eisenstat (2018)
% 
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.

m = n*(n-1)/2;      % dimension of the impact matrix
k = n^2*p + n + m;  % dimension of states    

    % prior
atheta = zeros(k,1); Vtheta = 10*ones(k,1);
nu0 = 5*ones(n,1); S0 = ones(n,1).*(nu0-1);
alp0 = 2*ones(r,1);  % sysmetric prior

cpri = -.5*k*r*log(2*pi) - .5*r*sum(log(Vtheta)) + nu0'*log(S0) - sum(gammaln(nu0));
prior = @(the,s,pr) cpri -.5*(the-repmat(atheta,r,1))'*((the-repmat(atheta,r,1))./repmat(Vtheta,r,1)) ...
    -(nu0+1)'*log(s) - sum(S0./s) + sum(ldiripdf(pr,alp0)) ;

    % compute and define a few things
tmpY = [Y0(end-p+1:end,:); shortY];
X = zeros(T,n*p); 
for i=1:p
    X(:,(i-1)*n+1:i*n) = tmpY(p-i+1:end-i,:);
end
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for j=2:n
    X2(j:n:end,count+1:count+j-1) = -shortY(:,1:j-1);
    count = count + j-1;
end
bigX = [SURform2([ones(T,1) X],n) sparse(X2)];

    % initialize the Markov chain
S = [kron((1:r-1)',ones(floor(T/r),1));r*ones(T-floor(T/r)*(r-1),1)];
P = triu(.2*ones(r,r)/(r-1),1) + tril(.2*ones(r,r)/(r-1),-1);
P = P + .8*eye(r);

theta = zeros(k,r);
sumE2 = zeros(n,1);
for i=1:r
    idx = (S == i);
    Ti = sum(idx);
    Z = [ones(Ti,1) X(idx,:)];
    shortYi = shortY(idx,:);
    tmptheta = (Z'*Z)\(Z'*shortYi);
    E = shortYi - Z*tmptheta;
    theta(1:n^2*p+n,i) = reshape(tmptheta',n^2*p+n,1);
    sumE2 = sumE2 + sum(E.^2)';
end
Sig = sumE2/T;

% initialize for storage
store_Sig = zeros(nsims,n); 
store_theta = zeros(nsims,k*r);
store_P = zeros(nsims,r,r);
store_S = zeros(T,r);
like = zeros(T,r);
tmpP1 = zeros(T,r);                   % p(s_t|Y_t,\theta,P)
tmpP2 = zeros(T,r); tmpP2(1,:) = 1/3; % p(s_t|Y_{t-1},\theta,P)


%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp(['Starting VAR-RS-R2 with ' num2str(r) ' regimes.... ']);
start_time = clock;

for isim = 1:nsims + burnin    
    % sample theta
    sumE2 = zeros(n,1);
    for i=1:r
            % extract data for state S_t == i
        idx = (S == i);
        Ti = sum(idx);
        if Ti == 0 % not an active regime, draw from prior
            theta(:,i) = atheta + sqrt(Vtheta).*randn(k,1);  
            Sig = 1./gamrnd(nu0,1./S0);
        else           
            shortYi = shortY(idx,:);
            Yi = reshape(shortY(idx,:)',Ti*n,1);
        
            X2i = zeros(n*Ti,n*(n-1)/2);
            count = 0;
            for j=2:n
                X2i(j:n:end,count+1:count+j-1) = -shortY(idx,1:j-1);
                count = count + j-1;
            end
            bigXi = [SURform2([ones(Ti,1) X(idx,:)],n) sparse(X2i)];                
            XiSig = bigXi'*sparse(1:Ti*n,1:Ti*n,repmat(1./Sig,Ti,1));
            Ktheta = sparse(1:k,1:k,1./Vtheta) + XiSig*bigXi;
            theta_hat = Ktheta\(atheta./Vtheta + XiSig*Yi);
            theta(:,i) = theta_hat + chol(Ktheta,'lower')'\randn(k,1);  
            
            sumE2 = sumE2 + sum((reshape(Yi - bigXi*theta(:,i),n,Ti)').^2)';            
        end
    end
    
        % sample Sig
    Sig = 1./gamrnd(nu0+T/2, 1./(S0 + sumE2/2));
    
        % sample S    
    for i=1:r
        mu_i = bigX*theta(:,i);        
        like(:,i) = mvnpdf(shortY,reshape(mu_i,n,T)',Sig');
    end 
    for t=1:T
        if t>1
            tmpP2(t,:) = tmpP1(t-1,:)*P;  
        end
        tmpP1(t,:) = tmpP2(t,:) .* like(t,:);        
        tmpP1(t,:) = tmpP1(t,:)/sum(tmpP1(t,:));
    end
    for t = T:-1:1        
        if t == T
            S(t) = find(cumsum(tmpP1(t,:)) > rand,1);
        else
            prob = tmpP1(t,:)'.*P(:,S(t+1));            
            prob = prob/sum(prob);
            S(t) = find(cumsum(prob) > rand,1);
        end
    end
    
        % sample P
    for i = 1:r
        ni = zeros(r,1);
        idx = find(S(1:end-1) == i);
        for j = 1:r
            ni(j) = sum(S(idx+1) == j);
        end       
        P(i,:) = dirirnd(alp0+ni);
    end
    
    if isim>burnin
        isave = isim-burnin;
        store_theta(isave,:) = theta(:);
        store_Sig(isave,:) = Sig(:);
        for j=1:r
            store_S(:,j) = store_S(:,j) + (S == j);
        end
        store_P(isave,:,:) = P;
    end
    
    if (mod(isim, 10000) == 0)
        disp([num2str(isim) ' loops... '])
    end 
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

theta_hat = mean(store_theta)';
thetaCI = quantile(store_theta,[.05 .95])';
S_hat = store_S/nsims;

