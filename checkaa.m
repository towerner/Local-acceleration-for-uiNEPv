function [H,gamma,shift,maxiter,tol,m,init] = checkaa(nepvpar,aapar)
% Check input structs for AA variants and return relevant parameters
try H = nepvpar.H;
catch
    error('H not supplied')
end
try gamma = nepvpar.gamma;
catch
    error('gamma not supplied')
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
try maxiter = aapar.maxiter;
catch
    maxiter = 100;
end
try tol = aapar.tol;
catch 
    tol = 1e-6;
end
try m = aapar.m;
catch
    m = maxiter;
end
try init = aapar.initsize;
catch
    init = 1;
end
end