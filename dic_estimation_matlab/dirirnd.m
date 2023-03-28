function draws = dirirnd(alp,N)

if nargin == 1
    N = 1;
end
n = length(alp);
x = gamrnd(repmat(alp',N,1),1,N,n);
draws = x./repmat(sum(x,2),1,n);
end

