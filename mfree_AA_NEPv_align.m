function [x,res,iter] = mfree_AA_NEPv_align(v0,nepvpar,accpar)
%[x,res,iter] = mfree_AA_NEPv_align(v0,nepvpar,accpar)
% -------------------------------------------------------------------------
% Solve unitarily invariant NEPv H(V)V=VD via SCF with Anderson
% acceleration applied to the aligned basis [[V]] w.r.t. a full rank Z
% This version assumes that H is only accessible via matvecs
% Alg. 3 from Lu/Werner 
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
%           fixed point residuals ||g([[Vi]])-[[Vi]]||
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
% Check if MATLAB version is recent, if tensorprod is not available we need
% to use for-loop. Same result, less efficient
usetp = exist('tensorprod','builtin');
iscmplx = ~isreal(v0);
% Setup eigensolver
eigsopt = {'Tolerance',max(1.1*eps,min(1e-14,tol/10)),...
        'FailureTreatment','keep','SubspaceDimension',max(2*k,30),...
        'MaxIterations',max(n,500),'IsFunctionSymmetric',true};
% Define matrix Z=Valign for alignment, orthogonal
rng(0);
Valign = randn(n,k,'like',v0);
Valign = orth(Valign);

% Get x0 from alignment
v0 = align(v0,Valign); % This is x0
%% Do first Fix-point step to obtain g0=g(x0) and f0=g0-x0 
gamma0 = gammafun(v0);
Hx = Hfun(gamma0,v0);
vold = v0;
% Compute NEPv-residual 
d0 = v0'*Hx; err = Hx-v0*d0;
res(1,1:2) = [norm(err,'fro'),toc(startaa)];
fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,1,res(1,1:2))
% Solve linear EVP to get x1
[v0,~] = eigs(@(v)Hfun(gamma0,v),n,k,shift,eigsopt{:},'StartVector',v0(:,1));
gamma0 = gammafun(v0);      % gamma(v0)=gamma(g0), g0 = [[v0]]
g0 = align(v0,Valign);
f0 = g0-vold;               % This is f0
res(1,3) = norm(f0,'fro');  % FP residual
v0 = g0;                    % This is x1
% First FP-step complete, setup AA basis in mode 1
Xs = zeros(n,k,m); Fs = zeros(n,k,m); xindx = 1;
Xs(:,:,xindx) = v0-vold;   % This is Delta(x0) 
tupdate = 0;
%% Main AA-loop
for ii=2:maxiter
    vold = v0; fold = f0; 
    % Compute NEPv-residual
    Hx = Hfun(gamma0,v0);
    d0 = v0'*Hx; err = Hx-v0*d0;
    res(ii,1:2) = [norm(err,'fro'),toc(startaa)];
    fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,ii,res(ii,1:2))
    % Solve linear EVP for SCF
    [v0,~] = eigs(@(v)Hfun(gamma0,v),n,k,shift,eigsopt{:},'StartVector',v0(:,1));
    g0 = align(v0,Valign);      % This is g(xi)
    f0 = g0-vold;               % This is f(xi)
    res(ii,3) = norm(f0,'fro'); % FP residual
    % Check for convergence
    if res(ii,1)<tol 
        break;
    end
    % AA-mixing:
    Fs(:,:,xindx) = f0-fold;  % This is fi-fim1=Delta(fim1) 
    if ii>=init
        s = tic;
        % Solve least squares problem, vectorized, split real and imaginary
        % part if necessary to get real alpha
        if iscmplx
            FF = reshape(Fs(:,:,1:xindx),n*k,xindx);
            ff = f0(:);
            alpha = [real(FF);imag(FF)]\[real(ff);imag(ff)]; 
        else
            alpha = reshape(Fs(:,:,1:xindx),n*k,xindx)\f0(:);
        end
        % Update v0. Carefull, straight sum does not necessarily preserve
        % orthogonality of v0!
        % Mix differences first
        if usetp
            dx = tensorprod(Fs(:,:,1:xindx)+Xs(:,:,1:xindx),-alpha,3,1);
            v0 = g0 + dx;           % Straight sum, regard as matrices
        else
            % For older versions of MATLAB
            v0 = g0;
            for ll=1:xindx
                v0 = v0 - alpha(ll).*(Fs(:,:,ll)+Xs(:,:,ll));
            end
        end
        % Obtain orthogonal iterate by projecting
         v0 = orth(v0);
        % Align to get next iterate
        v0 = align(v0,Valign);
        tupdate = tupdate + toc(s);
    else
        % First steps, just do simple FP
        v0 = g0;
    end
    % Update difference and truncate if necessary
    xindx = xindx + 1;
    Xs(:,:,xindx) = v0-vold; % This is xip1-xi
    if xindx>m
        Xs(:,:,1) = []; Fs(:,:,1) = [];
        xindx = m;
    end
    gamma0 = gammafun(v0);
end
x = v0;
iter = ii;
res = res(1:iter,:);
fprintf('Additional time for AA-mixing: %6.3f seconds.\n',tupdate)
end        

%% Subroutines used by mfree_AA_NEPv_align
% Basis alignment of V w.r.t. Valign, V=V*Q, where Q is the orthogonal
% Polar factor of V'*Valign computed via SVD
function V = align(V,Valign)
[U,~,W] = svd(V'*Valign,'econ','vector');
% Make sure to compute (U*W') first, both are kxk
V = V*(U*W');
end

