% Reproduces Fig. 5

clear
close all
initsize = 2;   
export = false;

% Multi-View clustering via Uniform Multidimensional scaling(UMDS), 
% three different data sets
sets = {'citeseer','cora','reuters1500'};
m=4; tol=1e-14; usenewton = true;   
maxgmres = 25; maxmr = 40;
for set = sets
    switch set{:}
        case 'citeseer'
            maxit = 750;        %n = 3312, s = 2 
        case 'cora'
            maxit = 500;        %n = 2708, s = 4  
        case 'reuters1500'
            maxit = 550;        %n = 1500, s = 5 
    end

    % Setup NEPv        
    nepvpar = setup_multiview_umds(set{:});
    if isempty(nepvpar)
        return;
    end      
    pname=sprintf('Multi-View-Clustering-UMDS, %s Data set',set{:});
    fname=sprintf('plots/mvc_umds_%s',set{:});
    
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
    [Xs,Rs] = mfree_callsolvers(X0,nepvpar,accpar,gntpar,paras);
    drawnow    
end


