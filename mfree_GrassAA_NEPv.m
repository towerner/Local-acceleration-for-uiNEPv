function [x,res,iter] = mfree_GrassAA_NEPv(v0,nepvpar,accpar)
%[x,res,iter] = mfree_GrassAA_NEPv(v0,nepvpar,accpar)
% -------------------------------------------------------------------------
% Solve unitarily invariant NEPv H(V)V=VD via SCF with Grassmannian 
% Anderson acceleration applied to eigenbasis Vi 
% This version assumes that H is only accessible via matvecs
% Alg. 4 from Lu/Werner 
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
%           fixed point residuals ||g(Vi)-Vi||
% iter:     required AA steps
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% Start timer and check parameters
startaa = tic;
if nargin<1 || isempty(v0) || ~isnumeric(v0)
    error('Initial guess needs to be supplied to GrassAA')
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

%% Do first Fix-point step to obtain g0=g(x0) and f0=g0-x0 
% Get gamma0 from evaluating gamma(v0)
gamma0 = gammafun(v0);
% Compute NEPv-residual 
Hx = Hfun(gamma0,v0);
d0 = v0'*Hx; err = Hx-v0*d0;
res(1,1:2) = [norm(err,'fro'),toc(startaa)];
fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,1,res(1,1:2))
% Solve linear EVP to get x1
[g0,~] = eigs(@(v)Hfun(gamma0,v),n,k,shift,eigsopt{:},'StartVector',v0(:,1));
% Grassmannian AA: v0=g(vold), 
% Replace difference f0=g0-x0 by Riemannian Logarithm 
% We have v0 = g(vold), i.e., r0=-f(vold)=Log_vold(v0)
r0 = grasslog(v0,g0);           % This is r0 in the tangent space
res(1,3) = norm(r0,'fro');      % FP residual in tangent space
% Get v1 by retracting r0, same subspace as g0 so just use that
%[vnew,UdH,SdH,VdH] = retract(r0,v0); 
vnew = g0;                      % This is x1
% New gamma
gamma0 = gammafun(vnew);

% First FP-step complete, setup AA basis, mode-1
Xs = zeros(n,k,m); Fs = zeros(n,k,m); xindx = 1;
% Get DX_0, already computed before
% Xs(:,:,1) = grasslog(v0,vnew);          % This is Delta(x0)
Xs(:,:,1) = r0;
tupdate = 0;
%% Main GrassAA-loop:
for ii=2:maxiter
    % We have rold = r(k-1), vold = v(k-1), v0=v(k)
    rold = r0; vold = v0; v0 = vnew;
    % Compute NEPv-residual
    Hx = Hfun(gamma0,v0);
    d0 = v0'*Hx; err = Hx-v0*d0;
    res(ii,1:2) = [norm(err,'fro'),toc(startaa)];
    fprintf('Step %*d, Residual norm %12.8e, total computation time %5.2f seconds\n',order,ii,res(ii,1:2))
    % Solve linear EVP for SCF
    [g0,~] = eigs(@(v)Hfun(gamma0,v),n,k,shift,eigsopt{:},'StartVector',v0(:,1));
    % Compute rk in the Tangent space
    r0 = grasslog(v0,g0);           % This is f(xk) in the tangent space
    res(ii,3) = norm(r0,'fro');     % FP residual in tangent space
    % Check for convergence
    if res(ii,1)<tol
        v0 = g0;
        break;
    end
    
    % AA mixing:
    % Precompute SVD of transport direction, multiple vector transports 
    % will be necessary. Should only be necessary in the first steps 
    % (before first retraction is applied) as SVD is known as byproduct of 
    % retraction afterwards
    s = tic;
    if ~exist('UdH','var') 
        dH = Xs(:,:,xindx);
        [UdH,SdH,VdH] = svd(dH,'econ','vector');
    end
    % Compute Delta r(k-1) in the tangent space
    Fs(:,:,xindx) = r0 - vectortransport(rold,vold,UdH,SdH,VdH);    
    % Vector Transport for all other increments
    Xs(:,:,1:xindx) = vectortransport(Xs(:,:,1:xindx),vold,UdH,SdH,VdH);
    Fs(:,:,1:xindx-1) = vectortransport(Fs(:,:,1:xindx-1),vold,UdH,SdH,VdH); 
    tupdate = tupdate + toc(s);
    % Now, all increments are in the Tangent space of v0 and we can use
    % basic linear algebra:
    if ii>=init
        s = tic;
        % Solve least squares problem, split real and imaginary part if 
        % necessary to get real alpha
        if iscmplx
            FF = reshape(Fs(:,:,1:xindx),n*k,xindx);
            rr = r0(:);
            alpha = [real(FF);imag(FF)]\[real(rr);imag(rr)]; 
        else
            alpha = reshape(Fs(:,:,1:xindx),n*k,xindx)\r0(:);
        end
        
        if usetp
            % Compute temporary rk on the Tangent space:
            rtemp = r0 - tensorprod(Fs(:,:,1:xindx),alpha,3,1);
            % rtemp = rtemp - v0*(v0'*rtemp); %Optional projection
            % Compute dX(k) in the Tangent space
            dx = rtemp - tensorprod(Xs(:,:,1:xindx),alpha,3,1);
        else
            % For older versions of MATLAB:
            rtemp = r0;
            for ll=1:xindx
                rtemp = rtemp - alpha(ll).*Fs(:,:,ll);
            end
            % rtemp = rtemp - v0*(v0'*rtemp); %Optional projection
            dx = rtemp;
            for ll=1:xindx
                dx = dx - alpha(ll).*Xs(:,:,ll);
            end
        end
        % Make sure direction dx is in the Tangent space of v0 
        dx = dx - v0*(v0'*dx);          %Optional
        % Retract to stay on the manifold
        [vnew,UdH,SdH,VdH] = retract(dx,v0);
        tupdate = tupdate + toc(s);
    else
        % First steps, just do simple FP
        % Normal SCF step and difference on the manifold via log
        vnew = g0;
        dx = r0;
    end
    % Update difference and truncate if necessary
    xindx = xindx + 1;
    Xs(:,:,xindx) = dx; % This is xip1-xi
    if xindx>m
        Xs(:,:,1) = []; Fs(:,:,1) = [];
        xindx = m;
    end
    % Get new gamma
    gamma0 = gammafun(vnew);
    
end
x = v0;
iter = ii;
res = res(1:iter,:);
fprintf('Additional time for AA-mixing: %6.3f seconds.\n',tupdate)
end

%% Subroutines used by mfree_GrassAA_NEPv
% Vector transport of Z from X0 to X1, where the direction D=Log_X0(X1) is
% given in terms of its SVD D=USV'
function Z = vectortransport(Z,X0,U,S,V)
% The transport on the Grassmannian is given by 
% T(Z) = ((I-U*U') + U*cos(S)*U' - X0*V*sin(S)*U')*Z
% (see Edelman/Arias/Smith, 1998, "The Geometry of Algorithms with 
% Orthogonality Constraints", Eq. (2.68)) 
if isempty(Z)
    return;
end
% Transport in parallel, BLAS3
block = false;
nblock = size(Z,3);
if nblock>1
    block = true;
    [n,k] = size(X0);
    Z = reshape(Z,n,k*nblock);
end
W = Z - U*(U'*Z);
C = U'*Z;
c1 = cos(S).*C;
c2 = (V.*sin(S)')*C;
Z = U*c1-X0*c2;
Z = Z + W;
% Reshape to 3D tensor
if block
    Z = reshape(Z,n,k,nblock);
end
end

% Riemannian logarithm on the Grassmannian
function F = grasslog(X,Y)
% Slight modification of one SVD-version of Alg.1 from Grassmann-Handbook,
% available from GitHub
% https://github.com/RalfZimmermannSDU/RiemannGrassmannLog/tree/main

%Y = orth(Y);  % Guarantee that Y is a point on the Stiefel manifold
% Compute SVD of Y'X
M = Y'*X;
[U,S,V] = svd(M,'econ','vector');
% Flip matrices for reordering
U = flip(U,2);
V = flip(V,2);
S = min(flip(S),1.0);

% Multiply by U and project
Y = Y*U;            
Y = Y - X*(X'*Y);    
% Replace sqrt(1-S^2) and asin from original version, which required
% reverse ordering, i.e. flipping all the matrices, by acos(S)/sin(acos(S))
% which is more efficient and mathematically equivalent
S = sqrt(1-S.^2);
theta = asin(S);
% If subspaces X and Y are close, we might have singular values of M close 
% to one, leading to singular values of the projected matrix close to 0.
% Since S/asin(S) can create NaN, we replace it by the limit, which is one
% x/asin(x) -> 1, x->0 
is0 = theta<100*eps;
theta(is0) = 1;
theta(~is0) = S(~is0)./theta(~is0);

% Form Z
V = V.*(theta');
F = Y*V';
end

% Retraction on the Grassmann manifold, full step along the geodesic 
% starting at x0 in direction of x
function [x,U,S,V] = retract(x,x0)
% Make sure that x is in the tangent space of x0 and compute its SVD
x = x - x0*(x0'*x);
[U,S,V] = svd(x,'econ',"vector");
% Compute Riemann exponential:
% We have R(x) = (x0*V*cos(S) + U*sin(S))*V'
% Since V is small and square and U is tall and skinny, we try to work on V
% as much as possible
v1 = (V.*cos(S)')*V';
v2 = V.*sin(S)';
x = x0*v1+U*v2';
end