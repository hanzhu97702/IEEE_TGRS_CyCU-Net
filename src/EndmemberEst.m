function P = EndmemberEst(Y,X, maxiter)

epsilon = 1e-6;
iter=0;

[l,N]=size(Y);
[k,N]=size(X);

P=zeros(l,k);

lamda1=zeros(size(P));

stop = false;
mu=1e-3;
rho=1.5;
mu_bar=1e6;

while ~stop && iter < maxiter+1
    
    iter=iter+1;
    A = (Y * X' + mu * P +lamda1)/(X*X'+mu * eye(size(X*X')));
    P=max(A-lamda1/mu,0); 
    lamda1=lamda1+mu*(P-A);
    mu=min(mu*rho,mu_bar);
    r_P=norm(P-A,'fro');

    if r_P<epsilon
            stop = true;
            break;
    end
end

end