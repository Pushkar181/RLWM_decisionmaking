Ms = [];

%% RLWM - classic RLWM model family

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 0 0 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
curr_model.pMin = [0 -1 0 0 0];
curr_model.pMax = [1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM0 RL0 0';

Ms{1}=curr_model;
options = optimoptions('fmincon','Display','off');
niter = 10000;

fit_model = Ms{1};
pmin = fit_model.pMin;
pmax = fit_model.pMax;

k=1;
fitmeasures = [];
fitparams = [];
expe_data = readmatrix('labdata7.xlsx');
X = expe_data;
data = X;
sofar= [];
j=0;
m=1;
npar = length(curr_model.pMin);
pars = repmat(curr_model.pMin, niter, 1) + rand(niter, npar) .* repmat(curr_model.pMax - curr_model.pMin, niter, 1);
for K=2:5
    % Define the objective function
    myfitfun = @(p) RLWM(p,K, data, fit_model);
    % Optimization loop
    for it = 1:niter
        par = pars(it, :);
        j=j+1;
        [p, fval, exitflag, output, lambda, grad, hessian] = ...
            fmincon(myfitfun, par, [], [], [], [], curr_model.pMin, curr_model.pMax, [], options);
        % Store best likelihood and parameters
        sofar(j, :) = [p,K, fval];
        
    end
end

        %% store information

        % global minimum
[llh,i]=min(sofar(:,end));
param = sofar(i(1),1:end-1);
        % uncomment for debugging
%        [s param llh]
%        [m k length(subjects_list) toc]
ntrials = size(data,1);

        % compute AIC, BIC,
        % add one for capacity
AIC = 2*llh + 2*(length(param)+1);
BIC = 2*llh + log(ntrials)*(length(param)+1);
AIC0 = -2*log(1/3)*ntrials;
psr2 = (AIC0-AIC)/AIC0;

        % store fit measures and best fit params
fitmeasures(k,:) = [-llh AIC BIC psr2 AIC0];
fitparams(k,:) = param;

    % store for the models in a new folder to not overwrite my fits.
All_Params{m} = fitparams;
All_fits(:,:,m) = fitmeasures;
save(['FitRLWM_datasetLab7','Ms','All_Params','All_fits'])