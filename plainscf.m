function [V,res,iter] = plainscf(V0,scfpars,solver)
%[V,res,iter] = plainscf(V0,scfpars,solver)
% -------------------------------------------------------------------------
% Self-Consistent Field Iteration for solving NEPv 
%           H(V)V=V D   s.t. V'V=I, D=D'
% by iteratively computing k eigenpairs of H(Vi) 
% This version assumes that H is feasible to store
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Inputs:
% V0:       initial eigenbasis V0, nxk matrix with orthonormal columns
% scfpar:   struct containing the parameter setup for the SCF algorithm
% solver:   Use direct (eig) or iterative (eigs) eigensolver
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
[H,shift,maxiter,tol] = checkscf(scfpars);
if nargin<3 || isempty(solver)
    solver='eigs';
end
order = floor(log10(maxiter))+1;
res = zeros(maxiter,2);
% Setup eigensolver if iterative
if strcmp(solver,'eigs')
    eigsopt = {'Tolerance',max(1.1*eps,min(1e-14,tol/10)),...
            'FailureTreatment','keep','SubspaceDimension',max(2*k,30),...
            'MaxIterations',max(500,n)};
end
% Compute initial H(V)
HV = H(V0);

%% Main SCF-Loop ----------------------------------------------------------
for ll=1:maxiter
    % Compute NEPv-residual 
    HVV = HV*V0;
    D0 = V0'*(HVV); D0 = 1/2.*(D0+D0');
    err = HVV-V0*D0;
    res(ll,:) = [norm(err,'fro'),toc(startscf)];
    fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,ll,res(ll,:))
    % Check for convergence
    if res(ll,1)<tol
        break;
    end

    % Solve linear EVP 
    if ~ishermitian(HV)
        HV = 1/2.*(HV+HV');
    end
    switch solver
        case 'eigs'
            [V0,~] = eigs(HV,k,shift,eigsopt{:},'StartVector',V0(:,1));
        case 'eig'
            if issparse(HV)
                HV = full(HV);
            end
            [V0,d] = eig(HV,'vector');
            [V0,~] = geteigs(V0,d,shift,k);
        otherwise
            error('Invalid eigenvalue solver.')
    end
    % Evaluate H at new iterate 
    HV = H(V0);
    
end % for l
% ------------------------------------------------------------ End SCF-Loop
V = V0;
res=res(1:ll,:);
iter = ll;
end %function


