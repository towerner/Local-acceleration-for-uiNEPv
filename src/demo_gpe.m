% Reproduces Fig. 3

clear
close all
export = false;

% Gross-Pitaevskii-equation
rng(0);
N = 100; 
% N=200;    %only timings for larger problem
k = 1; 
pot = 'harmonic'; 
tol = 1e-14;
initsize = 1;

usenewton = true;
maxgmres = 100; maxmr = 100;
solver = 'eigs';
fname=sprintf('plots/modgpe_%s_N%d',pot,N);
% Try different betas, 1.5, 4 and 14
betas = [4.0 5.5 7.2];
maxit = 70;

m = 4;
for beta = betas
    % Set up problem
    [nepvpar,xdisc,ydisc] = setup_gpe(N,beta,pot);
    pname=sprintf('GPE-2D, N=%d, k=%d, beta=%4.2f, pot=%s',N,k,beta,pot);

    %% common...
    X0 = nepvpar.x0;
    [n,k] = size(X0);   
    disp('Problem setup complete')
    %% Run solvers 
     
    accpar = struct('tol',tol,'maxiter',maxit,'m',m, ...
        'shift',nepvpar.shift,'initsize',initsize);
    gntpar = struct('tol',tol,'maxiter',maxit,'maxgmres',maxgmres,'maxmr',maxmr);
    paras = struct('usenewton',usenewton,'export',export,'pname',pname,'fname',fname);
    % Call solvers:
    [Xs,Rs] = callsolvers(X0,nepvpar,accpar,gntpar,paras,solver);
    drawnow
end   