% Reproduces Fig. 4

clear
close all
usenewton = true;
export = false;
maxmr = [];

% Sparse PCA with (2,1)-norm regularization, two data sets
for set = {'usps','coil'}
    switch set{:}
        case 'coil' 
            k = 30; alpha = 1e-1; m=4; 
            tol = 1e-11; maxit = 200;
            initsize = 25;
            solver = 'eig';
        case 'usps'
            k = 50; alpha = 1e-1; m=4; 
            tol = 1e-11; maxit = 120;
            initsize = 10;
            solver = 'eig';
    end
    
    % Set up problem
    [nepvpar,eps0] = setup_pca(k,alpha,set{:});
    if isempty(nepvpar)
        return;
    end    
    pname=sprintf('Sparse-PCA, alpha=%2.1e, eps_0=%4.2e, set=%s',alpha,eps0,set{:});
    fname=sprintf('plots/pca_%s',set{:});

    %% common...
    X0 = nepvpar.x0;
    [n,k] = size(X0);
    maxgmres = ceil(1.1*(n+k)*k);
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
