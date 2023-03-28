function Xout = constructX(X,X2,n)
k = n*(n-1)/2;
[T,m] = size(X);
if T == 0
    T = size(X2,1);
    X1 = [];
    idi1 = [];
    idj1 = [];
    idj2 = (1:T*k)';
else
    X1 = [ones(n*T,1) kron(X,ones(n,1))];
    m = m+1;
    tempid = reshape(1:T*(m*n+k),m*n+k,T)';  
    idi1 = kron((1:n*T)',ones(m,1));
    idj1 = reshape(tempid(:,1:n*m)',T*m*n,1);
    idj2 = reshape(tempid(:,n*m+1:end)',T*k,1);
end
idi2 = zeros(T*k,1);
count = 0;
for i=2:n
    for j=1:i-1
        idi2(count+1:k:end) = i:n:T*n;
        count = count + 1;
    end   
end
Xout = sparse([idi1; idi2],[idj1; idj2],[reshape(X1',n*T*m,1);reshape(X2',T*k,1)]);
end