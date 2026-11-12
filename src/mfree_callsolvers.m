function [Xs,Rs] = mfree_callsolvers(X0,nepvpar,accpar,ntpar,paras)
% Call all solvers and plot results
k = size(X0,2); m=accpar.m; maxmr = ntpar.maxmr; 
tol = accpar.tol; maxit = accpar.maxiter; initsize = accpar.initsize;
usenewton = paras.usenewton;
% Define colors
myalign = [0.49400,0.18400,0.55600];
mygrassaa = [0.89804,0.45882,0.12157];
myscf = [1.00000,0.00000,1.00000];
mynt = [0.63500,0.07800,0.18400]; 
% Check if ui parameterization is available for this problem
try uipara = paras.uipara;
catch 
    uipara = false;
end

%% Call methods
solvers = {'SCF',sprintf('AA(%d)-X',m),sprintf('AA(%d)-[[X]]',m),sprintf('AA(%d)-Gr',m)};
% -------------------------- SCF ------------------------------------------
scfpar = nepvpar;
scfpar.tol = min(accpar.tol,ntpar.tol);
scfpar.maxiter = accpar.maxiter;
s=tic;
[xscf,resscf,iterscf] = mfree_plainscf(X0,scfpar);
tscf = toc(s); 
fprintf('Elapsed time for SCF     is %8.5f seconds.\n',tscf);
Xs.SCF = xscf;
Rs.SCF = resscf;
% --------------------- AA-NEPv on X --------------------------------------
s=tic;
[xV,resV,iterV] = mfree_AA_NEPv_X(X0,nepvpar,accpar);
tV = toc(s);
fprintf('Elapsed time for AA-X    is %8.5f seconds.\n',tV);
Xs.AAX = xV;
Rs.AAX = resV;
%% Following methods are new contributions
% --------------------- AA-NEPv on rho ------------------------------------
if uipara
    solvers = [solvers(1),{sprintf('AA(%d)-rho',m)},solvers(2:end)'];
    s = tic;
    [xrho,resrho,iterrho] = mfree_AA_NEPv_rho(X0,nepvpar,accpar);
    trho = toc(s);
    fprintf('Elapsed time for AA-rho  is %8.5f seconds.\n',trho);
    Xs.AArho = xrho;
    Rs.AArho = resrho;  
end
% ---------------- AA with basis alignment --------------------------------
s=tic;
[xalign,resalign,iteralign] = mfree_AA_NEPv_align(X0,nepvpar,accpar);
talign = toc(s);
fprintf('Elapsed time for Al-AA   is %8.5f seconds.\n',talign);
Xs.AAalign = xalign;
Rs.AAalign = resalign;
% ----------------- Grassmannian AA-NEPv ----------------------------------
s=tic;
[xgaa,resgaa,itergaa] = mfree_GrassAA_NEPv(X0,nepvpar,accpar);
tgaa = toc(s);
fprintf('Elapsed time for GrassAA is %8.5f seconds.\n',tgaa);
Xs.GAA = xgaa;
Rs.GAA = resgaa;
% ------------------ Newton Acceleration ----------------------------------
if usenewton
    solvers = [solvers(:)',{'AlignNT'}];
    ntpar.initsize = accpar.initsize;
    s=tic;
    [xnt,resnt,iternt,info] = mfree_NewtonAcceleration(X0,nepvpar,ntpar);
    tnt = toc(s);
    fprintf('Elapsed time for NTacc   is %8.5f seconds.\n',tnt);
    Xs.NT = xnt;
    Rs.NT = resnt;
end


%% Just plotting from here:  
figure('Position',[0 0 1600 1000])

% Res,Iter
subplot(1,3,1)
semilogy(resscf(:,1),'Color',myscf,'Marker','x','LineStyle','-')
hold on
semilogy(resV(:,1),'k+-')
if uipara
    semilogy(resrho(:,1),'bo-')   
end
semilogy(resalign(:,1),'Color',myalign,'Marker','o','LineStyle','-')
semilogy(resgaa(:,1),'Color',mygrassaa,'Marker','o','LineStyle','-')
if usenewton
    semilogy(resnt(:,1),'Color',mynt,'Marker','d','LineStyle','-')
end
legend(solvers{:},'Location','northeast');
xlabel('its')
xlim([0,iterscf+1])
ylabel('Res.norm')
title('Eigenvalue residual ||H(x)x-xd||')

% Res,Time
subplot(1,3,2)
semilogy(resscf(:,2),resscf(:,1),'Color',myscf,'Marker','x','LineStyle','-')
hold on
semilogy(resV(:,2),resV(:,1),'k+-')
if uipara
    semilogy(resrho(:,2),resrho(:,1),'bo-')
end
semilogy(resalign(:,2),resalign(:,1),'Color',myalign,'Marker','o','LineStyle','-')
semilogy(resgaa(:,2),resgaa(:,1),'Color',mygrassaa,'Marker','o','LineStyle','-')
if usenewton
    semilogy(resnt(:,2),resnt(:,1),'Color',mynt,'Marker','d','LineStyle','-')
end
legend(solvers{:},'Location','northeast');
xlabel('Time[s]')
xlim([0,1.1*tscf])
ylabel('Res.norm')
title('Eigenvalue residual ||H(x)x-xd||')

% FP-Res,Iter
ixAA = contains(solvers,'AA');
subplot(1,3,3)
semilogy(resV(:,3),'k+-')
hold on
if uipara
    semilogy(resrho(:,3),'bo-')
end
semilogy(resalign(:,3),'Color',myalign,'Marker','o','LineStyle','-')
semilogy(resgaa(:,3),'Color',mygrassaa,'Marker','o','LineStyle','-')
legend(solvers{ixAA},'Location','northeast');
xlabel('its')
maxconv = max([iteralign,itergaa]);
if uipara
    maxconv = max([maxconv,iterrho]);
end
xlim([0,maxconv+1])
ylabel('Fixed-point norm')
title('Fixed-point residual ||g(x)-x||')
sgtitle(paras.pname)

% Compute costs, eigenvalues and angles between Eigenspaces 
f = nepvpar.cost;
symp = @(x) 1/2*(x+x');
rq = @(x) eig(symp(x'*(nepvpar.H(nepvpar.gamma(x),x))));
costs = [f(xscf),f(xV),f(xalign),f(xgaa)];  
angles = [0,subspace(xV,xscf),subspace(xalign,xscf),subspace(xgaa,xscf)];
times = [tscf,tV,talign,tgaa];
evs=[rq(xscf),rq(xV),rq(xalign),rq(xgaa)];
neigs = arrayfun(@(x) sprintf('Eigenvalue %d',x),1:k,'UniformOutput',false); 
if uipara
    costs = [costs(1),f(xrho),costs(2:end)];
    angles = [angles(1),subspace(xrho,xscf),angles(2:end)];
    times = [times(1),trho,times(2:end)];
    evs = [evs(1),rq(xrho),evs(2:end)];
end
if usenewton
    costs = [costs,f(xnt)];
    angles = [angles,subspace(xnt,xscf)];
    times = [times,tnt];
    evs = [evs,rq(xnt)];
end
    
tab = [costs;angles;times];

t1 = array2table(tab, 'VariableNames',solvers,'RowNames',{'Cost','Subspace Angles', 'Time [s]'});
disp(t1)
disp('-------------------------------------------------------------------')
t2 = array2table(evs, 'VariableNames',solvers,'RowNames',neigs);
disp(t2)


%% Just export data from here
if paras.export
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
    % Just residual-iter plot
    f1=figure();
    semilogy(0:(iterscf-1),resscf(:,1),'Color',myscf,'Marker','x','LineStyle','-')
    hold on
    semilogy(0:(iterV-1),resV(:,1),'k+-')
    if uipara
        semilogy(0:(iterrho-1),resrho(:,1),'bo-')
    end
    semilogy(0:(iteralign-1),resalign(:,1),'Color',myalign,'Marker','o','LineStyle','-')
    semilogy(0:(itergaa-1),resgaa(:,1),'Color',mygrassaa,'Marker','o','LineStyle','-')  
    if usenewton
        semilogy(0:(iternt-1),resnt(:,1),'Color',mynt,'Marker','d','LineStyle','-')
    end
    legend(solvers{:},'Location','northeast');
    xlabel('its')
    xlim([0,iterscf+1])
    ylabel('Res.norm')
    title('Eigenvalue residual ||H(x)x-xd||')
    % Export to tikz and eps
    if dotikz
        matlab2tikz('figurehandle',f1,'filename','residual_iter.tex');
    end
    print('residual_iter','-depsc2');
    % Just residual-time plot
    f2=figure();
    semilogy(resscf(:,2),resscf(:,1),'Color',myscf,'Marker','x','LineStyle','-')
    hold on
    semilogy(resV(:,2),resV(:,1),'k+-')
    if uipara
        semilogy(resrho(:,2),resrho(:,1),'bo-')
    end
    semilogy(resalign(:,2),resalign(:,1),'Color',myalign,'Marker','o','LineStyle','-')
    semilogy(resgaa(:,2),resgaa(:,1),'Color',mygrassaa,'Marker','o','LineStyle','-')   
    if usenewton
        semilogy(resnt(:,2),resnt(:,1),'Color',mynt,'Marker','d','LineStyle','-')
    end
    legend(solvers{:},'Location','northeast');
    xlabel('Time[s]')
    xlim([0,1.1*tscf])
    ylabel('Res.norm')
    title('Eigenvalue residual ||H(x)x-xd||')
    % Export to tikz and eps
    if dotikz
        matlab2tikz('figurehandle',f2,'filename','residual_time.tex');
    end
    print('residual_time','-depsc2');
    % Add problem information to be able to reproduce
    pinfo = struct('n',size(X0,1),'k',size(X0,2),'tol',tol,'maxit',maxit,'initsize',initsize);
    if exist("pot","var")
        pinfo.pot = pot;
    end
    if exist("beta","var")
        pinfo.beta = beta;
    end
    if isfield(nepvpar,'solver') && ~isempty(nepvpar.solver)
        pinfo.solver = nepvpar.solver;
    else
        pinfo.solver = 'eigs';
    end
    if isfield(nepvpar,'solveropts') && ~isempty(nepvpar.solveropts)
        pinfo.solveropts = nepvpar.solveropts;
    end
    if usenewton
        pinfo.maxgmres=ntpar.maxgmres;
        pinfo.maxmr=maxmr;
        pinfo.gmres=info;
    end
    warning off
    save('pdata.mat',"pinfo",'-mat')
    save('results.mat',"t1","t2",'-mat')
    close(f1)
    close(f2)
    cd(mypath)
    warning on
end
end