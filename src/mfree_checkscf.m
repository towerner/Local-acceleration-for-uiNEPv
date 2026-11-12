function [H,gamma,shift,maxiter,tol] = mfree_checkscf(scfpar)
% Check input structs for SCF and return relevant parameters
try H = scfpar.H;
catch
    error('H not supplied')
end
try gamma = scfpar.gamma;
catch
    error('gamma not supplied')
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
        shift = 'SR';
        warning(['SM not possible with user supplied function for A*x and use with eigs.' ...
            ' eigs expects A(x) to return A\x for shift=SM. Using SR instead.'])
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