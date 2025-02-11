function [sim_data, w_values, set_sizes] = RLWM(theta,expe_data,M)
% This function performs one simulation of the model, and returns simulated
% data in the same format as experimental data
% input
% theta: vector of parameters (size depends on model M)
% expe_data: experimental data matrix that constrains the experiment on
% which the model is simulated (e.g. sequence of blocks, stimuli, correct
% actions, etc.)
% M: structure containing information about model to be simulated.
%% define model parameters 
K = theta(end);% capacity
beta = 25;% softmax noise
alpha = theta(M.thetamapping(1));%learning rate
stick = theta(M.thetamapping(2));% motor perseveration
rho = theta(M.thetamapping(3));% WM weight
epsilon = theta(M.thetamapping(5));%lapse rate
forget = theta(M.thetamapping(4));%WM decay
% model dependent parameters
if M.pfixed(6)
    biasWM = M.fixedvalue(6);% WM learning rate bias
else
    biasWM = theta(M.thetamapping(6));
end
% RL learning rate bias
if M.pfixed(7)
    biasRL = M.fixedvalue(7);
else
    biasRL = theta(M.thetamapping(7));
end
% RL r0 value (r0=1 --> H model)
if M.pfixed(8)
    r0 = M.fixedvalue(8);
else
    r0 = theta(M.thetamapping(8));
end
% policy compression
if length(M.pfixed)<10
    chunk = 0;
elseif M.pfixed(9)
    chunk = M.fixedvalue(9);
else
    chunk = theta(M.thetamapping(9));
end
% set up learning rates [alpha- alpha+]
alphaRL = [alpha*biasRL alpha];
alphaWM = [biasWM 1];
if M.value==1
    or = [r0,1];
else
    or = [1,1];

end

%% set up the simulation experiment

% extract relevant experimental data
Allblocks = expe_data(:,3); %currentBlock
AllReward = expe_data(:,4); %reward obtained in action
AllPatch = expe_data(:,15); %total patches
AllwPat = expe_data(:,1); %selected patch

% create new simulation data
sim_data = expe_data;

% identify number of blocks 
blocks = unique(Allblocks)';

% number of actions
nA = 1;

w_values = [];
set_sizes = [];
%% start the simulation
% loop over blocks
for bl = blocks
    Tb = find(Allblocks == bl);
    % extract experiment information for current block
    stimuli = AllwPat(Tb);
    choices = AllwPat(Tb);
    rewards = AllReward(Tb);
    ns = AllPatch(Tb);
    ns=ns(1);
    Allpatchrew = expe_data(Tb,5:5+ns-1);

    % WM weight
    w = rho*min(1,K/ns);
    w_values = [w_values; w];
    set_sizes = [set_sizes; bl];
    if M.interact
        wint = w;
    else
        wint = 0;
    end
    
    % initialize RL/H weights, and WM weights.
    Q = (1/ns)*ones(nA,ns);
    WM = (1/ns)*ones(nA,ns);
    b= Q(1,:);
    %loop over trials - 
    for k = 2:length(choices)
        %s = stimuli(k-1);
        if k==2
            % make first choice same as participants
            choice = choices(k-1)+1;
        else
            % make a choice
            choice = select(b);
            % store choice
            choices(k-1) = choice-1;
        end
        % reward is deterministic if choice corresponds to target choice
        r = Allpatchrew(k-1,choice);
        % store reward
        rewards(k-1) = r;

        % model updates
        % WM decay
        WM = WM + forget*(1/ns - WM);
        % compute RL/H RPE
        if r==0
            rpe = or(r+1)-(wint*WM(1,choice) + (1-wint)*Q(1,choice));
        else 
            rpe = or(2)-(wint*WM(1,choice) + (1-wint)*Q(1,choice));
        end
        % update RL/H
        if r==0
            Q(1,choice) = Q(1,choice) + alphaRL(r+1)*rpe;
        else
            Q(1,choice) = Q(1,choice) + alphaRL(2)*rpe;
        end
        %update WM
        if r==0
            WM(1,choice) = WM(1,choice) + alphaWM(r+1)*(r-WM(1,choice));
        else
            WM(1,choice) = WM(1,choice) + alphaWM(2)*(r-WM(1,choice));
        end
        % set up sticky choice
        side=zeros(1,ns);
        side(choice)=1;
        
        % compute policy compression
        if chunk>0
            for s=1:ns
                W = Q(1,choice)+stick*side;
                bRLs(1,:) = exp(beta*W);
                bRLs(1,:) = epsilon/ns + (1-epsilon)*bRLs(1,:)/sum(bRLs(1,:));
                W = WM(1,choice)+stick*side;
                bWMs(1,:) = exp(beta*W);
                bWMs(1,:) = epsilon/ns + (1-epsilon)*bWMs(1,:)/sum(bWMs(1,:));
            end
            bs = w*mean(bWMs) + (1-w)*mean(bRLs);
        else
        bs=0;
        end
        
        % set up RL policy with sticky and compression
        W = Q(1,:)+stick*side+chunk*bs;
        bRL = exp(beta*W);
        % include random lapses
        bRL = epsilon/ns + (1-epsilon)*bRL/sum(bRL);
        % set up WM policy with sticky and compression
        W = WM(1,:)+stick*side+chunk*bs;
        bWM = exp(beta*W);
        % include random lapses
        bWM = epsilon/ns + (1-epsilon)*bWM/sum(bWM);
        % overall policy is the mixture
        b = w*bWM + (1-w)*bRL;
    end
    % store rewards and choices for this block
    Allrewards(Tb) = rewards;
    Allchoices(Tb) = choices;
    sim_data(Tb,4)=rewards;
    sim_data(Tb,1)=choices;

end
end

function a = select(pr)
% implement random choice selection based on a probability distribution
a = find([0 cumsum(pr)]<rand,1,'last');
end



runGenRec = true; % Set to true to re-run the generate and recover (SLOW)

if runGenRec
    clear all
    
    %% Load data and subjects and fit parameters
    gen_m = 1;
    datasets = {'labdata1.xlsx', 'labdata2.xlsx', 'labdata3.xlsx', 'labdata4.xlsx', 'labdata5.xlsx', 'labdata7.xlsx'};
    fit_files = {'FitRLWM_datasetLab1MsAll_ParamsAll_fits', 'FitRLWM_datasetLab2MsAll_ParamsAll_fits', ...
                 'FitRLWM_datasetLab3MsAll_ParamsAll_fits', 'FitRLWM_datasetLab4MsAll_ParamsAll_fits', ...
                 'FitRLWM_datasetLab5MsAll_ParamsAll_fits', 'FitRLWM_datasetLab7MsAll_ParamsAll_fits'};
    labels = {'Lab 1', 'Lab 2', 'Lab 3', 'Lab 4', 'Lab 5', 'Lab 7'};
    colors = lines(6); % Generate distinguishable colors for each dataset
    
    figure;
    hold on;
    
    all_unique_set_sizes = [];
    all_mean_w = [];
    all_std_w = [];
    
    for l = 6:6
        % Load dataset and corresponding fit file
        % q=l;
        % q
        load(fit_files{l});
        % q
        % l=q;
    %     l
        X = readmatrix(datasets{l});
        % l=q;
        gen_params = All_Params{gen_m};

        nsim = 3; % Number of iterations per participant
        all_w_values = {};
        all_set_sizes = {};

        for sim = 1:nsim
            [sim_data, w, sets] = RLWM(gen_params(1,:), X, Ms{gen_m});
            all_w_values{sim} = w;
            all_set_sizes{sim} = sets;
        end

        % Concatenate all w_values and set_sizes
        w_values = [all_w_values{:}];
        set_sizes = [all_set_sizes{:}];

        % Calculate mean and standard deviation of w_values for each unique set size
        unique_set_sizes = unique(set_sizes);
        mean_w = zeros(length(unique_set_sizes), 1);
        std_w = zeros(length(unique_set_sizes), 1);

        for i = 1:length(unique_set_sizes)
            current_set_size = unique_set_sizes(i);
            idx = set_sizes == current_set_size;
            ws = w_values(idx);
            mean_w(i) = mean(ws);
            std_w(i) = std(ws);
        end

        all_unique_set_sizes = [all_unique_set_sizes; unique_set_sizes];
        all_mean_w = [all_mean_w; mean_w];
        all_std_w = [all_std_w; std_w];

        % Plot WM contribution vs set size for this dataset
        % l=q;
        labels{l}
        errorbar(unique_set_sizes, mean_w, std_w, 'o-', 'Color', colors(l, :), 'DisplayName', labels{l});
    end

    xlabel('Block');
    ylabel('WM Contribution (w)');
    title('WM Contribution vs Set Size for All Datasets');
    legend show;
    hold off;
    % Plot actual vs simulated rewards
    figure;
    hold on;
    total_actual_rewards = sum(expe_data(:,4));
    total_simulated_rewards = sum(sim_data(:,4));

% Compute the difference
    reward_difference = total_actual_rewards - total_simulated_rewards;

% Display the results
    fprintf('Total Actual Rewards: %d\n', total_actual_rewards);
    fprintf('Total Simulated Rewards: %d\n', total_simulated_rewards);
    fprintf('Difference: %d\n', reward_difference);
    % Number of trials
    num_trials = size(expe_data, 1);
    x_values = 1:num_trials; % Sequence of choices

    % Plot actual rewards as blue dots
    plot(x_values, expe_data(:,4), 'b-', 'DisplayName', 'Actual Rewards');

    % Plot simulated rewards as red dots
    plot(x_values, sim_data(:,4), 'r-', 'DisplayName', 'Simulated Rewards');
    xlabel('Trial Number');
    ylabel('Reward');
    title('Actual vs Simulated Rewards');
    legend show;
    hold off;
end