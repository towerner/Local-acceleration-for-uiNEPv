% Reproduces Fig. 1 and Fig. 2

clear
close all
export = false;
compare_nt = true;
warning off

% Simple 1D KS Problem of size n, different choices of beta to see effect
% of stronger nonlinearity
N = 200; 
k = 2;
betas = [2 4 7]; 
maxit = 600;
betas = betas./(N^2);   % scale by problem size

tol = 1e-14;
initsize = 1;
m = 2;
solver = 'eig';

usenewton = true;
maxgmres = (N+k)*k; maxmr = [];
fname=sprintf('plots/ks1d');

% Setup for different betas
for beta = betas
    % Set up problem
    nepvpar = setup_ks1d(N,k,beta);
    
    %% common...
    X0 = nepvpar.x0;
    [n,k] = size(X0);
    pname=sprintf('KS-1D, N=%d, k=%d, beta=%3.1e',N,k,beta);
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
if compare_nt
    pause(1)
    % Reproduces Fig. 2 
    % Sensitivity of update equation close to solution
    x = Xs.AArho;
    gamma = nepvpar.gamma;
    H = nepvpar.H;
    dH = nepvpar.dH;
    
    %% Compute condition numbers
    Hx = H(gamma(x)); d = x'*(Hx*x); d = 1/2*(d+d');
    xx = [x;d];
    dHv = dH.setup(x);
    eigres = @(v,dv,lambda,dlambda) Hx*dv + dH.apply(v,dv,dHv) -(v*dlambda + dv*lambda);
    orthores = @(v,dv) -(v'*dv + dv'*v);
    subspaceres = @(v,dv) -v'*dv;
    dFx1 = @(y,dy) [eigres(y(1:n,:),dy(1:n,:),y(n+1:n+k,:),dy(n+1:n+k,:));...
         orthores(y(1:n,:),dy(1:n,:))]; % singular
    dFx2 = @(y,dy) [eigres(y(1:n,:),dy(1:n,:),y(n+1:n+k,:),dy(n+1:n+k,:));...
         subspaceres(y(1:n,:),dy(1:n,:))];  %non-singular
    E = eye((n+k)*k);
    KF1 = zeros((n+k)*k);
    KF2 = KF1;
    
    for ll=1:(n+k)*k
        Ei = E(:,ll);
        Ei = reshape(Ei,n+k,k);
        df1 = dFx1(xx,Ei); df2 = dFx2(xx,Ei);
        KF1(:,ll) = df1(:); KF2(:,ll) = df2(:);
    end
    k1 = cond(KF1); k2 = cond(KF2);
    %% Get convergence plots for GMRES
    rng(0);
    pertx = randn(n,k);     %"random" perturbation 
    testpert = 8:2:14; testpert = 10.^(-testpert);
    ntests = length(testpert);
    perts = cell(ntests,1);
    times = zeros(ntests,3);
    maxinner = (n+k)*k;
    fig=figure();
    for ll=1:ntests
        % Set up perturbed X and Lambda
        xtest = orth(x + testpert(ll).*pertx);
        Hx = H(gamma(xtest));
        dtest = xtest'*(Hx*xtest);
        % Set up RHS, F(X,Lambda)
        Fx = [Hx*xtest-xtest*dtest;zeros(k)];
        x0 = [xtest;dtest];
        % Define L_F for Newton(singular) and NTAcc(non-singular)
        dHx = dH.setup(xtest);
        eigres = @(v,dv,lambda,dlambda) Hx*dv + dH.apply(v,dv,dHx) -(v*dlambda + dv*lambda);
        orthores = @(v,dv) -(v'*dv + dv'*v);
        subspaceres = @(v,dv) -v'*dv;
        dFx1 = @(x,dx) [eigres(x(1:n,:),dx(1:n,:),x(n+1:n+k,:),dx(n+1:n+k,:));...
             orthores(x(1:n,:),dx(1:n,:))]; % singular
        dFx2 = @(x,dx) [eigres(x(1:n,:),dx(1:n,:),x(n+1:n+k,:),dx(n+1:n+k,:));...
             subspaceres(x(1:n,:),dx(1:n,:))];  %non-singular
        % Condition number
        for ii=1:(n+k)*k
            Ei = E(:,ii);
            Ei = reshape(Ei,n+k,k);
            df1 = dFx1(x0,Ei); df2 = dFx2(x0,Ei);
            KF1(:,ii) = df1(:); KF2(:,ii) = df2(:);
        end
        k1 = [k1,cond(KF1)]; k2 = [k2,cond(KF2)];
        % Sylvester operator for MINRES preconditioner
        Ax = [Hx,-xtest;-xtest',zeros(k)]; 
        Sx = @(y) Ax*y - y*dtest;
        M = @(y) MINRES(Sx,y,tol/10,maxmr);
        % Run GMRES for singular problem, for non-singular problem and for
        % non-singular problem with preconditioning
        [XNT,~,~,resNT] = FGMRES(@(y)dFx1(x0,y),-Fx,tol,maxinner);
        [XGNT,~,~,resGNT] = FGMRES(@(y)dFx2(x0,y),-Fx,tol,maxinner);
        [XPGNT,~,~,resPGNT] = FGMRES(@(y)dFx2(x0,y),-Fx,tol,maxinner,M);
        % Plot residuals
        subplot(1,3,1)
        semilogy(0:length(resNT)-1,resNT,'x','DisplayName',sprintf('diff=%1.0e',testpert(ll)))
        if ll==1
            hold on
            legend show
        end
        subplot(1,3,2)
        semilogy(0:length(resGNT)-1,resGNT,'x','DisplayName',sprintf('diff=%1.0e',testpert(ll)))
        if ll==1
            hold on
            legend show
        end
        subplot(1,3,3)
        semilogy(0:length(resPGNT)-1,resPGNT,'x','DisplayName',sprintf('diff=%1.0e',testpert(ll)))
        if ll==1
            hold on
            legend show
        end
        drawnow
        t1 = @()FGMRES(@(y)dFx1(x0,y),-Fx,tol,maxinner);
        t2 = @()FGMRES(@(y)dFx2(x0,y),-Fx,tol,maxinner);
        t3 = @()FGMRES(@(y)dFx2(x0,y),-Fx,tol,maxinner,M);
        times(ll,1) = timeit(t1,4);
        times(ll,2) = timeit(t2,4);
        times(ll,3) = timeit(t3,4);
        perts{ll} = sprintf('epsilon=1e%d',log10(testpert(ll)));
    end
    subplot(1,3,1)
    title('Newton-GMRES')
    subplot(1,3,2)
    title('Mod-Newton-GMRES')
    subplot(1,3,3)
    title('Mod-Newton-FGMRES')
    if export
        if ~isfolder("plots")
            mkdir("plots/")
        end
        if exist("matlab2tikz.m",'file')
            dotikz = true;
        else
            dotikz = false;
        end
        dname = sprintf('%s_%s',paras.fname,string(datetime)); 
        mkdir(dname)
        mypath = pwd;
        cd(dname)
        if dotikz
            matlab2tikz('figurehandle',fig,'filename','GMRES_RES.tex');
        end
        print('GMRES_RES','-depsc2');
        cd(mypath)
    end
    fprintf('\nComparison of timings and condition number for singular and non-singular correction equation:\n\n')
    T = array2table([k1(2:end);k2(2:end);times'],'VariableNames',perts,'RowNames',{'cond(NT)','cond(AlignNT)','Time NT','Time AlignNT', 'Time AlignNT w/MINRES'});
    disp(T)
end



