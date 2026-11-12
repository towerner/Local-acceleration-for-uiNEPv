function [x,res,iter] = mfree_AA_NEPv_rho(v0,nepvpar,accpar)
%[x,res,iter] = mfree_AA_NEPv_rho(v0,nepvpar,accpar)
% -------------------------------------------------------------------------
% Solve unitarily invariant NEPv H(V)V=VD via SCF with Anderson
% acceleration applied to a unitarily invariant parametrization gamma(V)
% This version assumes that H is only accessible via matvecs
% Alg. 2.2 from Lu/Werner/Bai
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Inputs:
% v0:       initial eigenbasis v0, nxk matrix with orthonormal columns
% nepvpar:  parameters related to uiNEPv, such as H, gamma,...
% accpar:   parameters for AA algorithm, such as convergence tolerance,
%           truncation parameter, eigenvalue solver,...
% -------------------------------------------------------------------------
% Outputs:
% x:        approximate eigenbasis of H
% res:      vector of residual norms over time, first column contains NEPv
%           residuals ||H(Vi)Vi-Vi*Di||, second column contains 
%           corresponding total computation time, third column contains
%           fixed point residuals ||g(rhoi)-rhoi||
% iter:     required AA steps
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% Start timer and check parameters
startaa = tic;
if nargin<1 || isempty(v0) || ~isnumeric(v0)
    error('Initial guess needs to be supplied to AA')
end
[n,k] = size(v0);
[Hfun,gammafun,shift,maxiter,tol,m,init] = mfree_checkaa(nepvpar,accpar);
order = floor(log10(maxiter))+1;
res = zeros(maxiter,3);
% Setup eigensolver
eigsopt = {'Tolerance',max(1.1*eps,min(1e-14,tol/10)),...
        'FailureTreatment','keep','SubspaceDimension',max(2*k,30),...
        'MaxIterations',max(n,500),'IsFunctionSymmetric',true};

% Get x0 from evaluating gamma
x0 = gammafun(v0); % This is x0
iscmplx = ~isreal(x0);
%% Do first Fix-point step to obtain g0=g(x0) and f0=g0-x0 
Hx = Hfun(x0,v0);       
% Compute NEPv-residual 
d0 = v0'*Hx; err = Hx-v0*d0;
res(1,1:2) = [norm(err,'fro'),toc(startaa)];
fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,1,res(1,1:2))
% Solve linear EVP to get x1
[v0,~] = eigs(@(v)Hfun(x0,v),n,k,shift,eigsopt{:},'StartVector',v0(:,1));
g0 = gammafun(v0);
f0 = g0-x0;             % This is f0
res(1,3) = norm(f0);    % FP residual
xold = x0;
x0 = g0;                % This is x1
% First FP-step complete, setup AA basis
szx = size(x0,1);
Xs = zeros(szx,m); Fs = zeros(szx,m); xindx = 1;
Xs(:,xindx) = x0-xold;  % This is Delta(x0) 
tupdate = 0;
%% Main AA-loop:
for ii=2:maxiter
    xold = x0; fold = f0;
    % Compute NEPv-residual
    Hx = Hfun(x0,v0);
    d0 = v0'*Hx; err = Hx-v0*d0;
    res(ii,1:2) = [norm(err,'fro'),toc(startaa)];
    fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,ii,res(ii,1:2))
    % Solve linear EVP for SCF
    [v0,~] = eigs(@(v)Hfun(x0,v),n,k,shift,eigsopt{:},'StartVector',v0(:,1));
    g0 = gammafun(v0);      % This is g(xi)
    f0 = g0-xold;           % This is f(xi)
    res(ii,3) = norm(f0);   % FP residual
    % Check for convergence
    if res(ii,1)<tol 
        break;
    end

    % AA mixing:
    Fs(:,xindx) = f0-fold;  % This is fi-fim1=Delta(fim1) 
    if ii>=init
        s = tic;
        % Solve least squares problem, split real and imaginary part if 
        % necessary to get real alpha
        if iscmplx
            alpha = [real(Fs(:,1:xindx));imag(Fs(:,1:xindx))]\[real(f0);imag(f0)];
        else
            alpha = Fs(:,1:xindx)\f0;
        end
        % Mix differences
        x0 = g0 - (Fs(:,1:xindx)+Xs(:,1:xindx))*alpha;
        tupdate = tupdate + toc(s);
    else
        % First steps, just do simple FP
        x0 = g0;
    end
    % Update difference and truncate if necessary
    xindx = xindx + 1;
    Xs(:,xindx) = x0-xold; % This is xip1-xi
    if xindx>m
        Xs(:,1) = []; Fs(:,1) = [];
        xindx = m;
    end    
end
x = v0;
iter = ii;
res = res(1:iter,:);
fprintf('Additional time for AA-mixing: %6.3f seconds.\n',tupdate)
end        
