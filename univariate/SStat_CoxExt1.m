function [stats,st] = SStat_CoxExt1(sID_ext,X_ext,d_ext,t_ext,e)
% [stats,st] = SStat_CoxExt1(sID_ext,X_ext,d_ext,t_ext,e)
%
% Parameter estimation for the extended Cox model. This function uses as
% input the output of SStat_X_ext with X_ext possibly modified by the user.
%
% Input
% sID_ext: Extended subjects' IDs.
% X_ext: Extended design matrix.
% d_ext: Extended censorship status vector.
% t_ext: Extended survival time vector.
% e: Convergence epsilon (gradient's norm). Default 10^-3;
%
% Output
% stats.Bhat: Estimated vector of the population regression parameters.
% stats.CovBhat: Estimated covariance matrix of the population regression 
% parameters.
% stats.llh: Values of the maximum log-likelihood across the optimization 
% process.
% st: Termination state (1 for convergence and 0 otherwise).
%
% $Revision: 1.1.1.1 $  $Date: 2013/27/03 11:25:52 $
% Original Author: Jorge Luis Bernal Rusiel 
% CVS Revision Info:
%    $Author: jbernal$
%    $Date: 2013/27/03 11:25:52 $
%    $Revision: 1.1 $
% References: Kleinbaum, D.G., Klein, M., 2005. Survival analysis. A self-
% learning approach, second edition. New York: Springer..
%   
if nargin < 4
    error('Too few inputs');
elseif nargin < 5
    e = 0.001;
end;
tic;
[n,p] = size(X_ext);
if (length(sID_ext)~=n) || (length(d_ext)~=n) || (length(t_ext)~=n)
    error(['All, the design matrix X_ext, the censorship status vector d_ext, the'...
        ' time vector t_ext and the subject ID vector must have the same number of rows.']);
end;
%indices of unique failure times in ft_ix (last index when ties happen)
st_ix = find(d_ext==1);
[~,ft_ix] = unique(t_ext(st_ix),'last');
ft_ix = st_ix(ft_ix);
%Starting values
Bhat = zeros(p,1);

%% Iterations
nit = 50;
gnorm = e+1;
it = 1;
display('Starting Newton-Raphson iterations');
while (gnorm>e) && (it<=nit)    
    gr = SStat_Gradient(X_ext,t_ext,Bhat,ft_ix);
    He = SStat_Hessian(X_ext,t_ext,Bhat,ft_ix);
    if (cond(He) < 1e+10)
        invHe = He\eye(p);
    else
        [Vtemp,Dtemp] = eig(He);
        invHe = Vtemp*diag(1./max(diag(Dtemp),1e-5))*Vtemp';
    end
    Bhat = Bhat - invHe*gr;
    %log-likelihood
    llh = SStat_Likelihood(X_ext,t_ext,Bhat,ft_ix);
    display(['Likelihood at iteration ' num2str(it) ' : ' num2str(llh)]);
    gnorm = norm(gr);
    display(['Gradient norm: ' num2str(gnorm)]);     
    it = it+1;
end;  
%% Termination
stats = struct('Bhat',Bhat,'CovBhat',-invHe,'llh',llh);
if (gnorm<=e)
    st = 1;
else
    st = 0;
    display(['Algorithm does not converge after ' num2str(nit)...
        ' iterations!!!']);
end;
et = toc;
display(['Total elapsed time is ' num2str(et) ' seconds']);
end






%% Likelihood, Gradient and Hessian

function llk = SStat_Likelihood(X_ext,t_ext,Bhat,ft_ix)
% 
% Log-likelihood value.
%
% Input
% X_ext: Extended design matrix.
% t_ext: Extended survival time vector.
% Bhat: Estimated vector of the population regression parameters.
% ft_ix: Failure time indices in X_ext (last index if any tie).
%
% Output
% llk: Log-likelihood value.
%
llk = 0;
nft = length(ft_ix);
for j=1:nft
    term = sum(exp(X_ext(t_ext(ft_ix(j))==t_ext,:)*Bhat));
    llk = llk + X_ext(ft_ix(j),:)*Bhat-log(term);
end;
end


function gr = SStat_Gradient(X_ext,t_ext,Bhat,ft_ix)
% 
% Gradient vector for the log-likelihood.
%
% Input
% X_ext: Extended design matrix.
% t_ext: Extended survival time vector.
% Bhat: Estimated vector of the population regression parameters.
% ft_ix: Failure time indices in X_ext (last index if any tie).
%
% Output
% gr: Gradient vector.
%
p = size(X_ext,2);
gr = zeros(p,1);
nft = length(ft_ix);
for j=1:nft
    riskset = t_ext(ft_ix(j))==t_ext;
    term = exp(X_ext(riskset,:)*Bhat); 
    gr = gr + (X_ext(ft_ix(j),:) - (term'*X_ext(riskset,:))/sum(term))';
end;
end


function He = SStat_Hessian(X_ext,t_ext,Bhat,ft_ix)
% 
% Hessian matrix for the log-likelihood.
%
% Input
% X_ext: Extended design matrix.
% t_ext: Extended survival time vector.
% Bhat: Estimated vector of the population regression parameters.
% ft_ix: Failure time indices in X_ext (last index if any tie).
%
% Output
% He: Hessian matrix.
%
p = size(X_ext,2);
He = zeros(p,p);
nft = length(ft_ix);
for j=1:nft
    rsk_ix = find(t_ext(ft_ix(j))==t_ext);  
    m = length(rsk_ix);
    term1 = 0;
    for i=1:m
        term1 = term1 + exp(X_ext(rsk_ix(i),:)*Bhat)*X_ext(rsk_ix(i),:)'*X_ext(rsk_ix(i),:);
    end;
    term2 = exp(X_ext(rsk_ix,:)*Bhat)'*X_ext(rsk_ix,:);
    term3 = sum(exp(X_ext(rsk_ix,:)*Bhat));
    He = He - (term1/term3-(term2'*term2)/(term3*term3));
end;
end

