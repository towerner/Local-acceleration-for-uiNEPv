function [x,res,iter,gmresinfo] = mfree_NewtonAcceleration(x0,nepvpar,ntpar)
%[x,res,iter,gmresinfo] = mfree_NewtonAcceleration(x0,nepvpar,ntpar)
% -------------------------------------------------------------------------
% Solve unitarily invariant NEPv H(V)V=VD via SCF with local Newton 
% acceleration
% This version assumes that H is only accessible via matvecs
% Alg. 5 from Lu/Werner 
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Inputs:
% x0:       initial eigenbasis v0, nxk matrix with orthonormal columns
% nepvpar:  parameters related to uiNEPv, such as H, gamma,...
% ntpar:    parameters for Newton algorithm, such as convergence tolerance,
%           GMRES parameters, eigenvalue solver,...
% -------------------------------------------------------------------------
% Outputs:
% x:        approximate eigenbasis of H
% res:      vector of residual norms over time, first column contains NEPv
%           residuals ||H(Vi)Vi-Vi*Di||, second column contains 
%           corresponding total computation time
% iter:     required AlignNT steps
% gmresinfo:GMRES convergence information per step 
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% Start timer and check parameters
startgnt = tic;
if nargin<1 || isempty(x0) || ~isnumeric(x0)
    error('Initial guess needs to be supplied to NTAcc')
end
[n,k] = size(x0);
[Hfun,gammafun,dH,iscmplx,shift,maxiter,tol,maxgmres,maxmr,initsize] ...
 = mfree_checknt(nepvpar,ntpar,n,k);
order = floor(log10(maxiter))+1;
res = zeros(maxiter+1,2);
% Setup eigensolver
eigsopt = {'Tolerance',max(1.1*eps,min(1e-14,tol/10)),...
        'FailureTreatment','keep','SubspaceDimension',max(2*k,30),...
        'MaxIterations',max(n,500),'IsFunctionSymmetric',true};
% Parameters for adaptive Eisenstat/Walker forcing 
ntol = 1/3; 
% Return GMRES convergence info?
showinfo = false;
if nargout>3
    showinfo = true;
    gmresinfo = zeros(maxiter,3);
end
% Enable flexible preconditioning by maxmr steps of global MINRES
if isempty(maxmr)
    useprecon = false;
else
    useprecon = true;
end
usent = false;
% Get first gamma(V)
gamma0 = gammafun(x0);
Hxx = Hfun(gamma0,x0); D = x0'*Hxx; D = 1/2.*(D+D');
% Compute initial NEPv-residual 
Ri = Hxx - x0*D;
resdg = norm(Ri,'fro');
res(1,:) = [resdg,toc(startgnt)];
%% Main NTAcc-loop
for ll=1:maxiter
    % Start with SCF-correction to guarantee we are looking for the correct
    % eigenvalues in D, Solve linear EVP 
    [x0,~] = eigs(@(v)Hfun(gamma0,v),n,k,shift,eigsopt{:},'StartVector',x0(:,1));
    gamma0 = gammafun(x0);
    Hx = Hfun(gamma0,x0);
    D = x0'*Hx; 
    % Force symmetry
    D = 1/2*(D+D');
    % Compute NEPv-residual 
    resdgold = resdg;
    Ri = Hx - x0*D;
    resdg = norm(Ri,'fro');
    res(ll+1,:) = [resdg,toc(startgnt)];
    fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,ll,res(ll+1,:))
    % Check for convergence
    if resdg<tol 
        break;
    end
 
    % Enable NT correction after initsize steps of SCF
    if ll==initsize
        usent = true;
    end

    % Setup solve of AlignNT correction equation
    if usent 
        
        % R-linear operator and right-hand side
        locA = @(Y) GMRESF(Y,@(x)Hfun(gamma0,x),@(dv)dH(x0,gamma0,dv),x0,D,x0);
        locB = [-Ri;zeros(k)];
        
        % Precondition by solving the symmetric Sylvester part using global
        % MINRES
        if useprecon
            locH = @(x) Hfun(gamma0,x(1:n,:));
            MINRESF = @(Z) [locH(Z) - x0*Z(n+1:n+k,:) - Z(1:n,:)*D; -x0'*Z(1:n,:) - Z(n+1:n+k,:)*D];
            mrtol = max(ntol/10,1e-14);
            M = @(Y) MINRES(MINRESF,Y,mrtol,maxmr);
            dX0 = M(locB);  % Initial guess by solving for symmetric part once    
        else
            M = [];
            dX0 = zeros(n+k,k);
        end
        
        % Solve by R-linear GMRES for complex problems and GMRES for real
        % problems when preconditioner is used, R-linear MINRES and MINRES
        % when no preconditioner is used
        ntol = innertol(resdgold,resdg,ntol);   % Forcing term
        if useprecon
            if iscmplx
                [dX,relres,initer] = FGMRES_RL(locA,locB,ntol,maxgmres,M,dX0);
            else
                [dX,relres,initer] = FGMRES(locA,locB,ntol,maxgmres,M,dX0);
            end
            % fprintf('Required %d steps of GMRES to achieve a relative residual of %12.8e where tolerance was %12.8e\n',initer,relres,ntol)
        else
            if iscmplx
                [dX,relres,initer] = MINRES_RL(locA,locB,ntol,maxgmres);
            else
                [dX,relres,initer] = MINRES(locA,locB,ntol,maxgmres);
            end
            fprintf('Required %d steps of MINRES to achieve a relative residual of %12.8e where tolerance was %12.8e\n',initer,relres,ntol)
        end
        % Store GMRES/MINRES convergence info if requested
        if showinfo
            gmresinfo(ll,:) = [ntol,relres,initer];
        end
        
        % Update X by Newton step, orthogonalize 
        x0 = x0 + dX(1:n,:);
        x0 = orth(x0);
        gamma0 = gammafun(x0);  
    end
end
iter = ll+1; res = res(1:iter,:);
x = x0;
if showinfo
    if iter<maxiter
        gmresinfo = gmresinfo(1:(iter-1),:);
    end
end
end

%% Subroutines used by AlignNT

% Compute adaptive tolerance for inexact Newton by Eisenstat/Walker
function eta = innertol(nrmfold,nrmf,etaold)
% Declare parameters
alpha = (1+sqrt(5))/2;
gamma = 0.9;
% Set safeguarding tolerances 
etamax = 1/3;
etamin = 0.1;
% Compute heuristic choice for new eta
eta = gamma*(nrmf/nrmfold)^alpha;
etapower = exp(((1+sqrt(5))/2)*log(etaold));
% Apply safeguarding
if etapower>etamin
    eta = max(eta,etapower);
end
eta = min(eta,etamax);
end

% Compute directional derivative in NTAcc
function dFX = GMRESF(y,Hx,dH,x,d,c)
% We have 
% F(x,d) = [H(x)*x-x*d; I-c'*x], where usually c=x, therefor 
% dF(x,d;yx,yd) = [H(x)*yx + dH(x,yx)*x - (x*yd + yx*d); -c'*yx];
% We implement dH such that it returns dH(x,yx)*x 
[n,k] = size(x);
% Split y according to x,d
yx = y(1:n,:); yd = y(n+1:n+k,:);
dHx = dH(yx);
dFX = [Hx(yx) + dHx - (x*yd + yx*d);-(c'*yx)];
end

