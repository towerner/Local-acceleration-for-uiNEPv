function [x0,res,iter,resvec] = MINRES_RL(A,B,acc,maxiter)
%[x0,res,iter,resvec] = MINRES_RL(A,B,acc,maxiter)
% -------------------------------------------------------------------------
% Computation of the solution of A*x=B via the global R-linear MINRES 
% Algorithm, where A is a complex R-linear operator which is self-adjoint 
% w.r.t. the real Frobenius inner product but potentially indefinite  
% -------------------------------------------------------------------------
% Modified version of Alg. T2.11 from 
% ITERATIVE METHODS FOR SINGULAR LINEAR EQUATIONS AND LEAST-SQUARES
% PROBLEMS, PhD-Thesis of Sou-Cheng Choi, Stanford, 2006 
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Inputs:
% A:        symmetric matrix or R-linear function for the evaluation of A*x 
% B:        right hand side matrix
% acc:      desired relative accuracy
% maxiter:  maximum number of steps
% -------------------------------------------------------------------------
% Outputs:
% x0:       approximate solution of A*x0=B up to accuracy acc
% res:      relative residual ||A*x-B||/||B|| of computed output x
% iter:     required MINRES steps
% resvec:   vector of relative residuals ||A*xi-B||/||B|| 
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% Allow matrix A (numeric) and function handle A(X) for evaluating A*X
if isnumeric(A)
    linfun = @(X) A*X;
else 
    linfun = A;
end
% Change inner product to get global R-linear MINRES
inprod = @(x,y) real(sum(conj(x).*y,'all'));

% Read problem size
[m,k] = size(B);
if nargin<4 || isempty(maxiter) 
    maxiter = min(100,2*m*k);    % set default for maxiter if not given
end
if nargin<3 || isempty(acc)
    acc = 1.0E-6;           	% set accuracy to 10^-6 if not given
end 
% Check for trivial solution
normb = norm(B,'fro');          
if normb==0                         % if right-hand side is zero, x0=0 
    x0 = zeros(m,k);                % immediately solves the equation
    res = 0;
    iter = 0;
    resvec = 0;
    return;
end
accb = acc*normb;                   % relative convergence tolerance
accb = max(accb,eps(normb));
storeres = false;

if nargout>3
    storeres = true;
    resvec = zeros(maxiter+1,1);
end

% Start MINRES
beta1 = normb; v0 = zeros(m,k); v1 = B./beta1;
phi0 = beta1; eps0 = 0;
delta1 = 0; 
c0 = -1; s0 = 0;
x0 = v0; d0 = v0; dm1 = v0;
if storeres
    resvec(1) = beta1;
end

for ii=1:maxiter
    % Extend sequence
    [alpha,beta2,vnew,flag] = lanczosstep(linfun,v1,v0,beta1,inprod);
    if flag
        break;
    end
    % Implicit QR
    delta2 = c0*delta1 + s0*alpha; gamma1 = s0*delta1 - c0*alpha;
    eps1 = s0*beta2; delta1 = -c0*beta2;
    [c1,s1,gamma2] = symgivens(gamma1,beta2);
    % Update implicit residual and x
    tau = c1*phi0; phi1 = s1*phi0; 
    if abs(gamma2)>1e-14
        dp1 = v1 - delta2.*d0 - eps0.*dm1; dp1 = dp1./gamma2;
        x0 = x0 + tau.*dp1; 
    else
        warning('Algorithm stagnating due to gamma=0 in step %d\n',ii)
        break;
    end
    res = abs(phi1);
    % Check for convergence
    if storeres
        resvec(ii+1) = res;
    end
    if res<accb 
        if nargout==1
            return;
        else
            break;
        end
    end
    % Copy memory
    beta1 = beta2; c0 = c1; s0 = s1;
    dm1 = d0; d0 = dp1; phi0  = phi1;
    v0 = v1; v1 = vnew; eps0 = eps1;

end
% Divide by ||B|| to get relative residuals
res = res/normb;
if storeres
    resvec = resvec(1:ii+1)./normb;
end
iter = ii;
end


%% Additional functions
function [alpha,beta,vkp1,flag] = lanczosstep(A,vk,vkm1,beta,inprod)
    % Extend Lanczos sequence by computing vkp1 = A*vk and orthogonalizing
    % against vk and vkm1
    % Essentially Alg. T2.1 from the same thesis
    flag = 0;
    vkp1 = A(vk);
    vkp1 = vkp1 - beta.*vkm1;
    alpha = inprod(vk,vkp1);
    vkp1 = vkp1 - alpha.*vk;
    beta = norm(vkp1,'fro');
    if beta>1e-14
        vkp1 = vkp1./beta;
    else
        warning('Beta=%12.8e in Lanczos!\n',beta)
        flag = -1;
    end
end


% Efficient and stable Givens rotations, Alg. T2.9 from the same thesis
function [c,s,r] = symgivens(a,b)
if b==0
    s = 0; r = abs(a); 
    if r==0
        c = 1;
    else
        c = sign(a);
    end
elseif a==0
    c = 0; s=sign(b); r = abs(b);
elseif abs(b)>abs(a)
    tau = a/b; s = sign(b)/sqrt(1+tau^2); c = s*tau; r = b/s;
elseif abs(a)>=abs(b)
    tau = b/a; c = sign(a)/sqrt(1+tau^2); s = c*tau; r = a/c;
end
end

