function [stats,st] = SStat_CoxPH1(X,d,t,e)
% [stats,st] = SStat_CoxPH1(X,d,t,e)
%
% Parameter estimation for the Cox proportional hazards model (Ties are not 
% handled).
%
% Input
% X: Design Matrix with the time-independent covariates. (mxp, m # of
% subjects, p # of covariates).  
% d: Logical vector (mx1) indicating censorship status (1 if the subject got 
% the failure event or 0 otherwise).
% t: Vector (mx1) whose entries are the survival and censored times (ordered 
% according to X).
% e: Convergence epsilon (gradient's norm). Default 10^-3;
%
% Output
% stats.Bhat: Estimated vector of the population regresion parameters.
% stats.CovBhat: Estimated covariance matrix of the population regresion 
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
if nargin < 3
    error('Too few inputs');
elseif nargin < 4
    e = 0.001;
end;
tic;
[m,p] = size(X);
if (length(d)~=m) || (length(t)~=m)
    error(['All, the design matrix X, the censorship status vector d and the'...
        ' time vector t must have the same umber of rows.']);
end;
%Sort the data by time. If there is a tie between a failure time and a
%censored time then the failure time goes first.
st_ix = find(d==1);
t1 = t(st_ix);
[t1,t1_ix] = sort(t1);
X1 = X(st_ix(t1_ix),:);
cs_ix = find(d==0);
if ~isempty(cs_ix)
    t2 = t(cs_ix);
    [t2,t2_ix] = sort(t2);
    X2 = X(cs_ix(t2_ix),:);
    count1 = 1; count2 = 1; i = 0;
    while (count1 <= length(t1)) && (count2 <= length(t2))
        i = i + 1;
        if t1(count1) <= t2(count2)
            X(i,:) = X1(count1,:);
            d(i) = 1;
            count1 = count1 + 1;
        else 
            X(i,:) = X2(count2,:);
            d(i) = 0;
            count2 = count2 + 1;
        end;
    end;
    if (count1 > length(t1))
        X(i+1:end,:) = X2(count2:end,:);
        d(i+1:end) = 0;
    else
        X(i+1:end,:) = X1(count1:end,:);
        d(i+1:end) = 1;
    end;
else
    X = X1;
end;
%Starting values
Bhat = zeros(p,1);

%% Iterations
nit = 50;
gnorm = e+1;
it = 1;
display('Starting Newton-Raphson iterations');
while (gnorm>e) && (it<=nit)    
    gr = SStat_Gradient(X,Bhat,d);
    He = SStat_Hessian(X,Bhat,d);
    if (cond(He) < 1e+10)
        invHe = He\eye(p);
    else
        [Vtemp,Dtemp] = eig(He);
        invHe = Vtemp*diag(1./max(diag(Dtemp),1e-5))*Vtemp';
    end  
    Bhat = Bhat - invHe*gr;
    %log-likelihood
    llh = SStat_Likelihood(X,Bhat,d);
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

function llk = SStat_Likelihood(X,Bhat,d)
% 
% Log-likelihood value.
%
% Input
% X: Ordered design Matrix (according to survival time).
% Bhat: Estimated vector of the population regression parameters.
% d: Logical vector indicating censorship status (1 if the subject got the 
% failure event or 0 otherwise).
%
% Output
% llk: Log-likelihood value.
%
m = size(X,1);
st_ix = find(d==1);
nft = sum(d);
llk = 0;
for i=1:nft
    term1 = sum(exp(X(st_ix(i):m,:)*Bhat));
    llk = llk + X(st_ix(i),:)*Bhat - log(term1);
end;
end


function gr = SStat_Gradient(X,Bhat,d)
% 
% Gradient vector for the log-likelihood.
%
% Input
% X: Ordered design Matrix (according to survival time).
% Bhat: Estimated vector of the population regression parameters.
% d: Logical vector indicating censorship status (1 if the subject got the 
% failure event or 0 otherwise).
%
% Output
% gr: Gradient vector.
%
[m,p] = size(X);
gr = zeros(p,1);
st_ix = find(d==1);
nft = sum(d);
for i=1:nft
    term = exp(X(st_ix(i):m,:)*Bhat);
    gr = gr + (X(st_ix(i),:) - (term'*X(st_ix(i):m,:))/sum(term))';
end;
end


function He = SStat_Hessian(X,Bhat,d)
% 
% Hessian matrix for the log-likelihood.
%
% Input
% X: Ordered design Matrix (according to survival time).
% Bhat: Estimated vector of the population regression parameters.
% d: Logical vector indicating censorship status (1 if the subject got the 
% failure event or 0 otherwise).
%
% Output
% He: Hessian matrix.
%
[m,p] = size(X);
st_ix = find(d==1);
nft = sum(d);
He = zeros(p,p);
for i=1:nft
    %numerator
    term1 = 0; 
    for j=st_ix(i):m
        term1 = term1 + exp(X(j,:)*Bhat)*X(j,:)'*X(j,:);
    end;
    term2 = exp(X(st_ix(i):m,:)*Bhat)'*X(st_ix(i):m,:);
    %denominator
    term3 = sum(exp(X(st_ix(i):m,:)*Bhat));
    He = He - (term1/term3-(term2'*term2)/(term3*term3));
end;
end

