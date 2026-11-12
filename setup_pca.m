function [nepvpar,eps0] = setup_pca(k,alpha,set)

% Set up sparse PCA-(2,1) problem, we have 
% H(V) = 2A - alpha*B(V)^(-1/2), where B(V) = diag(VV') + eps^2*I

% Regularization parameter eps0 and two data sets
eps0 = 1e-3;
switch set
    case 'coil'
        load data/PCA/COIL20.mat
    case 'usps'
        load data/PCA/USPS.mat
end
% Centralize data
X = X';
Xmean = mean(X,2);
X = X - Xmean;
[n,p] = size(X);
X = 1/sqrt(p-1) .*X;
     
% Compute sample covariance matrix A and optimal eps
eps0 = eps0 * sqrt(k/n);
% Parametrization via diagonal of B(V)^(1/2)
gammafun = @(x) sqrt(sum(x.*x,2)+eps0.^2);
% H(V) and Frechet derivative based on parametrization gamma
A = -2.*X*X';
Hfun = @(gamma) A + diag(alpha./gamma);
dHsetup = @(v) setupdH(v,eps0);
dHapply = @(v,dv,dHv) -alpha.*(sum(dv.*v,2)./dHv).*v; 
dHfun = struct('setup',dHsetup,'apply',dHapply);
% Initial value by largest eigenvalues of A=XX', i.e. largest singular
% values of X
[X,D] = eig(A,'vector');
shift = 'SR';
[X00,~] = geteigs(X,D,shift,k);
% Setup NEPv struct
f = @(x) -norm(X'*x)^2 +alpha.*sum(sqrt(sum(x.*x,2)+eps0^2));
nepvpar = struct('H',Hfun,'gamma',gammafun,'shift',shift, ...
        'x0',X00,'cost',f,'dH',dHfun,'iscmplx',false,...
        'Hscf',@(x)Hfun(gammafun(x)));
end   

function x = setupdH(v,eps0)
    % Precompute part of derivative (rho(v)+eps^2)^(3/2) independent of
    % direction
    x = sum(v.*v,2)+eps0^2;
    x = x.*sqrt(x);
end