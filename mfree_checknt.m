function [H,gamma,dH,iscmplx,shift,maxiter,tol,maxgmres,maxminres,init] = mfree_checknt(nepvpar,ntpar,n,k)
% Check input structs for NTAcc and return relevant parameters
try H = nepvpar.H;
catch
    error('H not supplied')
end
try iscmplx = nepvpar.iscmplx;
catch 
    warning('Assuming problem is complex. If the problem is real, please set ISCMPLX to false to allow for a more efficient solve.')
    iscmplx = true;
end
try gamma = nepvpar.gamma;
catch
    error('gamma not supplied')
end
try dH = nepvpar.dH;
catch
    error('dH no supplied')
end
try shift = nepvpar.shift;
catch 
    shift = 'SR';
    warning('shift not supplied, using SR')
end
switch upper(shift)
    case {'SR','SA','SMALLESTREAL'}
        shift = 'SR';
    case {'LR','LA','LARGESTREAL'}
        shift = 'LR';
    case {'LM','LARGESTABS'}
        shift = 'LM';
    case {'SM','SMALLESTABS'}
        shift = 'SM';
    otherwise
        warning('Shift must be one of SR, LR, LM or SM. Using default SR.')
end
try maxiter = ntpar.maxiter;
catch
    maxiter = 100;
end
try tol = ntpar.tol;
catch 
    tol = 1e-6;
end
try maxgmres = ntpar.maxgmres;
catch
    maxgmres = 100;
end
mmax = ceil(1.1*(n+k)*k);
if iscmplx
    mmax = 2*mmax;
end
maxgmres = min(maxgmres,mmax);
try maxminres = ntpar.maxmr;
catch
    maxminres = [];
end
maxminres = min(maxminres,n*k);
try init = ntpar.initsize;
catch
    init = 1;
end
end