function [H,shift,maxiter,tol] = checkscf(scfpar)
% Check input structs for SCF and return relevant parameters
try H = scfpar.Hscf;
catch
    error('H not supplied')
end

try shift = scfpar.shift;
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
try maxiter = scfpar.maxiter;
catch
    maxiter = 100;
end
try tol = scfpar.tol;
catch 
    tol = 1e-6;
end
end