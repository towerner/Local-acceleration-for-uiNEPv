function [V,res,iter] = mfree_plainscf(V0,scfpars)
%[V,res,iter] = mfree_plainscf(V0,scfpars)
% -------------------------------------------------------------------------
% Self-Consistent Field Iteration for solving NEPv 
%           H(V)V=V D   s.t. V'V=I, D=D'
% by iteratively computing k eigenpairs of H(Vi) 
% This version assumes that H is only accessible via matvecs
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Inputs:
% V0:       initial eigenbasis V0, nxk matrix with orthonormal columns
% scfpar:   struct containing the parameter setup for the SCF algorithm
% -------------------------------------------------------------------------
% Outputs:
% V:        approximate eigenbasis of H
% res:      vector of residual norms over time, first column contains NEPv
%           residuals ||H(Vi)Vi-Vi*Di||, second column contains 
%           corresponding total computation time
% iter:     required SCF steps
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% Start timer and check parameters
startscf = tic;
if nargin<1 || isempty(V0) || ~isnumeric(V0)
    error('Initial guess needs to be supplied to SCF')
end
[n,k] = size(V0);
[Hfun,gammafun,shift,maxiter,tol] = mfree_checkscf(scfpars);
order = floor(log10(maxiter))+1;
res = zeros(maxiter,2);
% Setup eigensolver
eigsopt = {'Tolerance',max(1.1*eps,min(1e-14,tol/10)),...
        'FailureTreatment','keep','SubspaceDimension',max(2*k,30),...
        'MaxIterations',max(n,500),'IsFunctionSymmetric',true};
% Compute initial gamma(V)
gamma0 = gammafun(V0);

%% Main SCF-Loop ----------------------------------------------------------
for l=1:maxiter
    % Compute NEPv-residual 
    HV = Hfun(gamma0,V0);
    D0 = V0'*HV;
    err = HV-V0*D0;
    res(l,:) = [norm(err,'fro'),toc(startscf)];
    fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,l,res(l,:))
    % Check for convergence
    if res(l,1)<tol
        break;
    end
    % Solve linear EVP 
    [V0,~] = eigs(@(v)Hfun(gamma0,v),n,k,shift,eigsopt{:},'StartVector',V0(:,1));
    % Evaluate gamma at new iterate 
    gamma0 = gammafun(V0);
    
end % for l
% ------------------------------------------------------------ End SCF-Loop
V = V0;
res=res(1:l,:);
iter = l;
end %function


