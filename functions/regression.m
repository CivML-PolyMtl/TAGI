%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         regression
% Description:  Main codes for UCI regression benchmarks
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      December 13, 2019
% Updated:      January 23, 2019
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [mp, Sp, metric, time, hplist] = regression(NN, x, y, trainIdx, testIdx)
% Initialization
initsv            = NN.sv;
initmaxEpoch      = NN.maxEpoch;
NN.errorRateEval  = 0;
% Indices for each parameter group
% Train net
NN.trainMode      = 1;  
NN.batchSize      = NN.batchSizelist(1);
NN                = indices.parameters(NN);
NN                = indices.covariance(NN);
% Validation net
NNval             = NN;
NNval.trainMode   = 0;  
NNval.batchSize   = NN.batchSizelist(2);
NNval             = indices.parameters(NNval);
NNval             = indices.covariance(NNval);
% Test net
NNtest            = NN;
NNtest.trainMode  = 0;
NNtest.batchSize  = NN.batchSizelist(3);
NNtest            = indices.parameters(NNtest);
NNtest            = indices.covariance(NNtest);    
% Loop
Nsplit            = NN.numSplits;
RMSElist          = zeros(Nsplit, 1);
LLlist            = zeros(Nsplit, 1);
trainTimelist     = zeros(Nsplit, 1);
hpOptTimelist     = zeros(Nsplit, 1);
hplist            = zeros(Nsplit, 2);
optHyperparameter = NN.optsv;
permuteData       = 0;
if isempty(trainIdx) || isempty(testIdx)
   permuteData    = 1; 
end
for s = 1:Nsplit
    disp('**********************')
    disp([' Run time #' num2str(s)])
    % Data
    if permuteData == 1
        [xtrain, ytrain, xtest, ytest] = dp.split(x, y, ratio);
    else
        xtrain = x(trainIdx{s}, :);
        ytrain = y(trainIdx{s}, :);
        xtest  = x(testIdx{s}, :);
        ytest  = y(testIdx{s}, :);
    end
    [xtrain, ytrain, xtest, ~, ~, ~, mytrain, sytrain] = dp.normalize(xtrain, ytrain, xtest, ytest);
    % Initalize weights and bias
    [mp, Sp] = tagi.initializeWeightBias(NN);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Hyperparameter tuning
    if optHyperparameter == 1 
        NN.sv           = initsv;
        NN.maxEpoch     = initmaxEpoch;
        numRuns         = NN.numFolds;
        tuningtime_s    = tic;
        optNumEpochs    = NN.optNumEpochs;
        Nepoch4hp       = 1;
        svlist          = zeros(Nepoch4hp, 1, NN.dtype);
        numEpochMaxlist = zeros(Nepoch4hp, 1, NN.dtype);
        for e = 1:Nepoch4hp
            disp(' ')
            disp(['  Opt. Epoch #' num2str(e) '/' num2str(Nepoch4hp) '|' num2str(s)])
            if e > 1               
                idxtrain = randperm(size(ytrain, 1));
                ytrain   = ytrain(idxtrain, :);
                xtrain   = xtrain(idxtrain, :);
            end
            [svlist(e), numEpochMaxlist(e)] = opt.crossValidation(NN, NNval, mp, Sp, xtrain, ytrain, optNumEpochs, numRuns);
        end
        sv               = mean(svlist);
        if optNumEpochs == 1
            maxEpoch     = round(mean(numEpochMaxlist));
        else
            maxEpoch     = NN.maxEpoch;
        end
        % Update hyperparameter values for network
        NN.sv            = sv;
        NN.maxEpoch      = maxEpoch;
        tuning_e         = toc(tuningtime_s);
        hpOptTimelist(s) = tuning_e;
        hplist(s, :)     = [NN.sv, NN.maxEpoch];
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Training
    NN.trainMode    = 1;
    rumtime_s       = tic;
    stop            = 0;
    epoch           = 0;
    while ~stop
        if epoch > 1
            idxtrain = randperm(size(ytrain, 1));
            ytrain   = ytrain(idxtrain, :);
            xtrain   = xtrain(idxtrain, :);
        end
        epoch = epoch + 1;
        [mp, Sp, ~, ~, ~] = tagi.network(NN, mp, Sp, xtrain, ytrain);
        if epoch >= NN.maxEpoch
            stop = 1;
        end
    end
    runtime_e = toc(rumtime_s);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Testing
    NNtest.sv                  = NN.sv;
    [~, ~, ynTest, SynTest, ~] = tagi.network(NNtest, mp, Sp, xtest, []);
    R                          = repmat(NNtest.sv.^2, [size(SynTest, 1), 1]);
    SynTest                    = SynTest + R;
    [ynTest, SynTest]          = dp.denormalize(ynTest, SynTest, mytrain, sytrain);   
    % Evaluation
    RMSElist(s)                = mt.computeError(ytest, ynTest);
    LLlist(s)                  = mt.loglik(ytest, ynTest, SynTest);
    trainTimelist(s)           = runtime_e;
    
    disp(' ')
    disp(['      RMSE : ' num2str(RMSElist(s)) ])
    disp(['      LL   : ' num2str(LLlist(s))])
    % Outputs
    metric.RMSElist    = RMSElist;
    metric.LLlist      = LLlist;
    time.trainTimelist = trainTimelist;
    time.hpOptTimelist = hpOptTimelist ;
end
end