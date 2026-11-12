function [x,res,iter,resvec] = FGMRES_RL(A,B,acc,maxiter,M,x)
%[x,res,iter,resvec] = FGMRES_RL(A,B,acc,maxiter,M,x)
% -------------------------------------------------------------------------
% Computation of the solution of A*x=B via the global R-linear flexible 
% GMRES Algorithm, we expect A to be a complex R-linear operator                   
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Inputs:
% A:        matrix or R-linear function for the evaluation of A*x 
% B:        right hand side matrix
% acc:      desired relative accuracy
% maxiter:  maximum number of steps
% M:        flexible preconditioner
% x:        initial starting guess for the iteration
% -------------------------------------------------------------------------
% Outputs:
% x:        approximate solution of A*x=B up to relative accuracy acc
% res:      relative residual ||A*x-B||/||B|| of computed output x
% iter:     required GMRES steps
% resvec:   vector of relative residuals in every step
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% Allow matrix A (numeric) and function handle A(X) for evaluating A*X
if isnumeric(A)
    linfun = @(X) A*X;
else
    linfun = A;
end
% Change inner product to get global R-linear GMRES
inprod = @(X,Y)real(sum(conj(X).*Y,'all'));

% Read problem size
[m,k] = size(B);
indx = 1:k;
%% Preliminary parameter Checks 
% Check for zero initial guess
iszero = false;
if nargin<6 || isempty(x)
    x = zeros(m,k);             % use the matrix of all zeros if no initial 
	iszero = true;			    % guess is given
elseif ~nnz(x)
    iszero = true;
end
% Check for problem consistency 
if ~isequal(size(x),[m,k]) 
    error('System is not consistent, dimension mismatch!');
end 
% Check for preconditioner
if nargin<5 || isempty(M)
    noprecon = true;
else
    noprecon = false;
end
% Check for maximum iterations
if nargin<4 || isempty(maxiter) 
    maxiter = min(100,2*m*k);     % set default for maxiter if not given
end
% Check for relative accuracy
if nargin<3 || isempty(acc)
    acc = 1.0E-6;           	% set accuracy to 10^-6 if not given
end 


% Initializations ---------------------------------------------------------
Z = zeros(maxiter+1,1);         % stores right hand sides for least 
			                    % squares solution
Hk = zeros(maxiter);            % stores the Hessenberg matrix
htemp = zeros(maxiter+1,1);     % one temporary column of Hk
V = zeros(m,k*(maxiter+1));     % stores the Arnoldi matrices in mode-1
if ~noprecon
    MV = zeros(m,k*maxiter);    % Preconditioned Arnoldi matrix
end
G = zeros(2,maxiter);           % stores the necessary Givens Rotations
% -------------------------------------------------------------------------
% Check for trivial solution
nrmb = norm(B,'fro');          
if nrmb==0                     % if right-hand side is zero, x0=0 
    x = zeros(m,k);             % immediately solves the equation
    res = 0;
    iter = 0;
    resvec = 0;
    return;
end
accb = acc*nrmb;   % relative convergence tolerance
accb = max(accb,eps(nrmb));
% Initial residual and residual norm
if iszero                   % If x=0, A*x=0, so R=B-A*x=B
    Res0 = B;
    beta = nrmb;
else                        % Else, R=B-A*x
    Res0 = B - linfun(x);                  
    beta = norm(Res0,'fro');               
end
% If initial residual satisfies tolerance, we immediately stop
if beta<accb
    res = beta/nrmb;
    iter = 0;
    resvec = res;
    return;
end
resvec = zeros(maxiter+1,1);
% Start Arnoldi
V(:,indx) = Res0./beta;             % initial Arnoldi matrix
Z(1) = beta;                        % initial LS parameters
resvec(1) = beta;

%% Start FGMRES-Loop ------------------------------------------------
for ll=1:maxiter
    % Start modified Arnoldi Algorithm ------------------------------------ 
    % Compute new Arnoldi-Matrix
    if noprecon                         % No flex preconditioning
        w = V(:,indx);
    else                                % With flex preconditioning
        w = M(V(:,indx));
        MV(:,indx) = w;
    end
    w = linfun(w);                      % W=A*V(l)
    % Orthogonalize against existing matrices w.r.t. R-linear inner product
    % <x,y> = real(trace(x'*y))
    orthindx = 1:k;
    for ii=1:ll
        htemp(ii) = inprod(w,V(:,orthindx));    % Modified inner product 
        w = w - htemp(ii).*V(:,orthindx);       % Orthogonalize
        orthindx = orthindx + k;
    end
    % Extend Arnoldi sequence by normalization
    htemp(ll+1) = norm(w,'fro');
    V(:,orthindx) = w./htemp(ll+1);

    % Apply old Givens Rotations to the l-th column of H (H is real)
    for ii = 1:(ll-1)
         tmpH = htemp(ii);
         htemp(ii) = G(1,ii)*tmpH + G(2,ii)*htemp(ii+1);
         htemp(ii+1) = G(1,ii)*htemp(ii+1) - G(2,ii)*tmpH;
    end  

    %Construct new Givens Rotation to Eliminate H(k+1,k) 
    t = norm(htemp(ll:ll+1));
    G(:,ll) = htemp(ll:ll+1)./t;
    htemp(ll) = t;
    htemp(ll+1) = 0;
    
    % Apply new Givens Rotation to Hk and Z
    Hk(1:ll,ll) = htemp(1:ll);
    Z(ll+1) = -G(2,ll).*Z(ll);
    Z(ll) = G(1,ll).*Z(ll);
    res = abs(Z(ll+1));              % store residual 
    resvec(ll+1) = res;
    % Check for convergence
    if res<=accb || ll == maxiter  
        iter = ll;
        break;                      
    end 
    indx = indx + k;
end  % end for ll
% -------------------------------------- End modified Arnoldi Algorithm
    
% Backward Substitution to compute the approximate solution 
Y = Hk(1:ll,1:ll)\Z(1:ll);
% Obtain solution x from linear combination of (preconditioned) Arnoldi 
% matrices and initial guess
indx = 1:k;
if noprecon
    for ii=1:ll
        x = x + Y(ii).*V(:,indx);
        indx = indx + k;
    end
else
    for ii=1:ll
        x = x + Y(ii).*MV(:,indx);
        indx = indx + k;
    end
end
% Divide by ||B|| to get relative residuals
res = res/nrmb;
resvec = resvec(1:ll+1)./nrmb;
% -------------------------------------------------- End GMRES_RL
end %function