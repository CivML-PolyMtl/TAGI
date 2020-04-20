%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         classification
% Description:  Main codes for classification task
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      December 18, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mp, Sp, metric, time, hplist, testIdx] = classification(NN, x, y, mp, Sp, trainIdx, testIdx)
% Initialization
NN.errorRateEval = 1;
NN.obsShow   = 10000; 
NN.task      = 'classification';
% Indices for each parameter group
% Train net
% Data
[xtrain, ytrain, trainLabels, trainEncoderIdx] = dp.selectData(x, y, NN.labels, NN.encoderIdx, trainIdx);
[xtest, ~, testLabels, ~] = dp.selectData(x, y, NN.labels, [], testIdx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyperparameter tuning
if NN.optsv == 1
    tuningtime_s    = tic;
    optNumEpochs    = NN.optNumEpochs;
    numEpochs4hp    = 1;
    numRuns         = 1;
    svlist          = zeros(numEpochs4hp, 1, NN.dtype);
    numEpochMaxlist = zeros(numEpochs4hp, 1, NN.dtype);
    optIdx          = randperm(size(ytrain, 1), 10000);
    [xopt, yopt, optlabels, optEncoderIdx] = dp.selectData(xtrain, ytrain, trainLabels, trainEncoderIdx, optIdx);
    NN.labels       = optlabels;
    NN.encoderIdx   = optEncoderIdx;
    for e = 1:numEpochs4hp
        disp(' ')
        disp(['  Opt. Epoch #' num2str(e) '/' num2str(numEpochs4hp)])
        if e > 1
            idxtrain = randperm(size(ytrain, 1));
            ytrain   = ytrain(idxtrain, :);
            xtrain   = xtrain(idxtrain, :);
        end
        [svlist(e), numEpochMaxlist(e)] = opt.crossValidation(NN, NN, mp, Sp, xopt, yopt, optNumEpochs, numRuns);
    end
    sv               = mean(svlist);
    if optNumEpochs == 1
        optEpoch     = round(mean(numEpochMaxlist));
    else
        optEpoch     = NN.maxEpoch;
    end
    % Update hyperparameter values for network
    NN.sv            = sv;
    NN.maxEpoch      = optEpoch;
    tuning_e         = toc(tuningtime_s);  
    hplist           = [NN.sv, NN.maxEpoch];
elseif NN.optNumEpochs == 1 && NN.optsv == 0 && strcmp(NN.task, 'classification') 
    disp(' ')
    disp('Optimizing the optimal number of epochs... ')
    tuningtime_s     = tic;
    NNval            = NN;
    NNval.trainMode  = 0;
    [xtrainHp, ytrainHp, trainHpLabels, trainHpEncoderIdx] = dp.selectData(xtrain, ytrain, trainLabels, trainEncoderIdx, [1:50000]);
    [xvalHp, yvalHp, valHpLabels, valHpEncoderIdx] = dp.selectData(xtrain, ytrain, trainLabels, trainEncoderIdx, [50000+1:60000]);  
    NN.labels        = trainHpLabels;
    NN.encoderIdx    = trainHpEncoderIdx;
    NNval.labels     = valHpLabels;
    NNval.encoderIdx = valHpEncoderIdx;
    optEpoch         = opt.numEpochs(NN.numEpochs, NN.sv, NN, NNval, mp, Sp, xtrainHp, ytrainHp, xvalHp, yvalHp);
    disp([' Num. of epoch: ' num2str(optEpoch)])
    NN.maxEpoch      = optEpoch;
    tuning_e         = toc(tuningtime_s);  
    hplist           = [NN.sv, NN.maxEpoch];
else
    tuning_e         = [];
    hplist           = [NN.sv, NN.maxEpoch];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
disp(' ')
disp('Training... ')
NN.trainMode    = 1;
NN.labels       = trainLabels;
NN.encoderIdx   = trainEncoderIdx;
NN.errorRateEval= 1;
runtime_s       = tic;
if NN.earlyStop == 1
    [mp, Sp, Mvallist, optEpoch] = opt.earlyStop(NN, mp, Sp, xtrain, ytrain);
    hplist(2)   = optEpoch;
else
    stop        = 0;
    epoch       = 0;   
    timeTot     = 0;
    while ~stop
        ts = tic;
        if epoch > 1
            idxtrain    = randperm(size(ytrain, 1));
            ytrain      = ytrain(idxtrain, :);
            xtrain      = xtrain(idxtrain, :);
            NN.labels   = NN.labels(idxtrain);
            NN.encoderIdx =  NN.encoderIdx(idxtrain, :);
        end
        epoch = epoch + 1;
        disp('#########################')
        disp(['Epoch #' num2str(epoch)])
        [mp, Sp, ~, ~, ~, ~] = tagi.network(NN, mp, Sp, xtrain, ytrain);
        if epoch >= NN.maxEpoch
            stop = 1;
        end
        timeTot         = timeTot + toc(ts);
        timeRem         = timeTot/epoch*(NN.maxEpoch-epoch)/60;
        if stop ~= 1
            disp(' ')
            disp(['Remaining training time (mins): ' sprintf('%0.2f',timeRem)]);
        end
    end
    Mvallist = nan;
end
runtime_e = toc(runtime_s);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
disp(' ')
disp('Testing... ')
NN.trainMode         = 0;
NN.labels            = testLabels;
NN.errorRateEval     = 1;
[~, ~, ~, ~, PnTest, erTest] = tagi.network(NN, mp, Sp, xtest, []);
metric.erTest        = erTest;
metric.PnTest        = PnTest;
metric.Mvallist      = Mvallist;
time.trainTimelist   = runtime_e;
time.hpOptTimelist   = tuning_e ;
end