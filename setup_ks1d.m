function nepvpar = setup_ks1d(n,k,beta)

% Set up 1D Kohn-Sham-Example, we have 
% H(V) = L + beta*diag(L\rho(V)),
% where L is a 1D Laplacian and rho(V)=diag(VV') is the charge density 
% vector 

% Use sparse matrices for NEPv
en = ones(n,1);
Lsparse = spdiags([-en,2*en,-en],-1:1,n,n);

% Efficient solve of Laplacian via Cholesky
[iL,~,iP] = chol(Lsparse,"lower");
LL = @(x)iP*(iL'\(iL\(iP'*x)));

% Parametrization via charge density
gammafun = @(X) sum(X.*X,2);
% H(V) and Frechet derivative based on parametrization gamma
Hfun = @(gamma) Lsparse + beta.*spdiags(LL(gamma),0,n,n); 
dHfun = struct('setup',@(x) (2*beta).*x, 'apply', @(v,dv,dHx) dhks(v,dv,LL,dHx));

% Eigenpairs of 1D Laplacian as initial guess
[X00,D] = eig(full(Lsparse),'vector');
[~,ix] = sort(D,'ascend');
X00 = X00(:,ix(1:k));

% Setup NEPv struct
shift = 'SR';
f = @(x) 1/2*sum(x.*(Lsparse*x),'all') + beta/4*gammafun(x)'*(LL(gammafun(x)));
nepvpar = struct('H',Hfun,'gamma',gammafun,'shift',shift, 'dH', dHfun,...
    'x0',X00,'Hscf',@(x)Hfun(gammafun(x)),'cost',f, 'iscmplx',false);
end    

% Frechet derivative of H
function dH = dhks(v,dv,LL,betav)
dg = sum(v.*dv,2); 
dH = LL(dg).*betav;
end
    