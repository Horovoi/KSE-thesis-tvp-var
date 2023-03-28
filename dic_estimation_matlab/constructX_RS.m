function bigX = constructX_RS(Y0,shortY,S,p)

tmpY = [Y0(end-p+1:end,:); shortY];
X = zeros(T,n*p); 
for i=1:p
    X(:,(i-1)*n+1:i*n) = tmpY(p-i+1:end-i,:);
end
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(i:n:end,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
X1 = SURform2([ones(T,1) X],n); 
bigX = [X1 sparse(X2)];
end