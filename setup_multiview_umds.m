function nepvpar = setup_multiview_umds(set)
% Set up UMDS-example, we have 
% H(V) = sum_j (Aj'VV'Aj), 
% where Aj are the affinity matrices of the data sets, symmetric matrices
store = false;  
% Choose between 3 different data sets
switch set
    case 'citeseer'
        name = 'data/UMDS/CiteSeer.mat';
        k = 6;
    case 'cora'
        name = 'data/UMDS/Cora.mat';
        k = 7;
    case 'reuters1500'
        name = 'data/UMDS/Reuters-1500.mat';
        k = 6;
end
% Load data sets
load(name) 
M = length(X); n = length(y); 
% Get pre-embedding and affinity matrices, if not supplied
fprintf('Setting up %s data set with %d views and %d samples,\n',set,M,n)
if ~exist('As','var') 
    store = true;
    As = zeros(n,n,M); en = ones(n,1); en = en./sqrt(n); 
    warning off
    for ii=1:M
        fprintf('Setting up pre-embedding for view %d...\n',ii)
        fprintf('Computing Affinity matrix... ')
        sX = sum(X{ii}.^2,2);
        Ds = sX + sX' - 2.*(X{ii}*X{ii}');
        Ds = 1/2.*(Ds+Ds');
        Ds = max(Ds,0);
        % Ds = exp(-Ds./2); %Gaussian kernel
        Ds = Ds - (Ds*en)*en';
        Ds = Ds - en*(en'*Ds);
        Ds = 1/2.*(Ds+Ds');
        Ds = -1/2.*Ds;
        nrmA = norm(Ds,'fro');
        As(:,:,ii) = Ds./nrmA;
        fprintf(' Done!\n') 
    end
end

% Define H and Frechet derivative, no unitarily invariant parametrization
% available, set gamma(:,:,i)=A_i*V such that 
% H(V)*x = sum_i gamma_i*(gamma_i'*x) low-rank

gammafun = @(x) Afun(x,As);
Hfun = @(gamma,v) evalH(gamma,v,M,n,k);
dHfun = @(v,gamma,dv) evaldH(gamma,v,dv,As);
if ~exist('sumAs','var')
    sumAs = sum(As,3);
    store = true;
end
% Initial eigenspace by largest eigenpairs of sum(Aj) (those are smallest of -sum(Aj))
rng(0);
[X00,~] = eigs(sumAs,k,'LR','Tolerance',eps,...
            'FailureTreatment','keep','SubspaceDimension',max(2*k,30),...
            'MaxIterations',max(n,500));
% Setup NEPv struct
shift = 'SR';
f = @(x) ffun(x,As);
nepvpar = struct('H',Hfun,'gamma',gammafun,'shift',shift, ...
        'x0',X00,'cost',f,'dH',dHfun,'iscmplx',false,'As',As);
if store
    save(name,"X","y","As","sumAs",'-v7.3')
end  
end

function Ax = Afun(x,As)
% Compute Ai*x, for all Ai in parallel, and store in vectorized form
Ax = pagemtimes(As,x);
Ax = Ax(:);
end

function Hx = evalH(gamma,x,M,n,k)
% Evaluate H(V)x efficiently, where H(V)=sum Ai'VV'Ai
% We have gamma(i) = Ai*V, such that 
% V'*Ai*x=gamma(i)'*x and 
% Ai'*V*V'*Aix=gamma(i)*gamma(i)'*x
% gamma is stored in vectorized form for use with AA-rho, reshape to block
% form
As = reshape(gamma,n,k*M);
% Apply in low-rank form
x = -x;
Hx = As*(As'*x);
end

function dHx = evaldH(gamma,x,dx,As)
% Evaluate Frechet derivative efficiently
M = size(As,3); [n,k] = size(x);
% Similar to evalH, reshape gamma to block form and apply in low-rank form
Asx = reshape(gamma,n,k*M);
% Need Aj*X, for all j, costly part
Asdx = pagemtimes(As,dx);
Asdx = reshape(Asdx,n,k*M);
% Apply in low-rank form
x = -x;
dHx = Asx*(Asdx'*x) + Asdx*(Asx'*x);
end

% Cost function, sum of norms of x'*Ai*x
function fx = ffun(x,As)
As = pagemtimes(As,x);
As = pagemtimes(x,'transpose',As,'none');
nrmAs = pagenorm(As,'fro').^2;
fx = -sum(nrmAs,3);
end