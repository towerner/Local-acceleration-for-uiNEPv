function [nepvpar,XX,YY] = setup_gpe(N,beta,pot)

% Set up 2D Gross-Pitaevskii-Example, we have
% H(v) = A + beta*diag(|v|^2), 
% where A is Hermitian and discretizes first and second order derivatives 
% as well as potential and beta>0
rng(0);
% set potential function
switch lower(pot)
    case 'harmonic'
        Vf = @(x,y) (x.^2+y.^2)/2;      	% symmetric potential 
    otherwise
        Vf = @(x,y) (x.^2+100*y.^2)/2;	% non-symmetric potential 
end
L = 1;          % domain [-L, L] * [-L, L]
h = 2*L/(N+1);  % stepsize
omega = 0.85;

% Construct A
DN = (diag(ones(N-1,1),1) - diag(ones(N-1,1),-1))/2/h; % central difference for 1st deriv
D2N = (diag(ones(N-1,1),1) -diag(ones(N,1))*2 +diag(ones(N-1,1),-1))/h/h; % for 2nd deriv
LN = kron(D2N,speye(N)) + kron(speye(N), D2N);
xx = linspace(-L, L, N+2); xx = xx(2:end-1)'; yy = xx;
LFN = kron(spdiags(xx, 0, N, N), DN) - kron(DN, spdiags(xx, 0, N, N));
[XX, YY] = meshgrid(xx,xx);
Vtilde = arrayfun(Vf, XX,YY);
Vtilde = Vtilde(:); % vectorize
A0tilde_real = -0.5*LN + spdiags(Vtilde, 0, N^2, N^2);
A0tilde_imag = -1*omega*LFN;
A0 = h^2*(A0tilde_real + 1i*A0tilde_imag);
A0 = 1/2.*(A0+A0');
n = N*N;

% Parametrization via |x|^2
gammafun = @(x) abs(x).^2;
% H(V) and Frechet derivative based on parametrization gamma
Hfun = @(gamma)  A0 + spdiags(beta.*gamma,0,n,n);
dHapply = @(v,dv,dHx) real(conj(v).*dv).*dHx;
dHfun = struct('setup',@(v)(2*beta).*v,'apply',dHapply);

% Initial guess, normalized complex all ones
x0 = ones(n,1)+1i.*ones(n,1);
x0 = x0./norm(x0);
% Setup NEPv struct
shift = 'SR';
f = @(x) 1/2*real(sum(conj(x).*(A0*x),'all')) + beta/4*real(sum(gammafun(x).^2)) ;
nepvpar = struct('H',Hfun,'gamma',gammafun,'shift',shift, ...
        'x0',x0,'Hscf',@(x)Hfun(gammafun(x)),'cost',f,'dH',dHfun, ...
        'iscmplx',true);
end  