%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         powerPlant
% Description:  Apply TAGI to power plant dataset
% Author:       Luong-Ha Nguyen & James-A. Goulet
% Created:      December 13, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
%% Data
path                     = char([cd ,'/data/']);
load(char([path, '/powerPlant.mat']))
load(char([path, '/powerPlantTestIndices.mat']))
load(char([path, '/powerPlantTrainIndices.mat']))
nobs                     = size(data,1);
ncvr                     = 4;
% Input features
x                        = data(:,1:ncvr);
% Output targets
y                        = data(:,end); 
nx                       = size(x, 2);        
ny                       = size(y, 2); 
%% Neural Network properties
% GPU 1: yes; 0: no
NN.gpu                   = 0;
% Data type object single or double precision
NN.dtype                 = 'single';
% Number of input covariates
NN.nx                    = nx; 
% Number of output responses
NN.ny                    = ny;   
% Batch size [train, val, test]
NN.batchSizelist         = [1 1 1]; 
% Number of nodes for each layer
NN.nodes                 = [NN.nx 50 NN.ny]; 
% Input standard deviation        
NN.sx                    = nan;
% Observations standard deviation
NN.sv                    = 0.24*ones(1, ny);        
% Maximal number of learnign epoch
NN.maxEpoch              = 40;   
% Factor for initializing weights & bias
NN.factor4Bp             = 1E-2*ones(1,numel(NN.nodes)-1);
NN.factor4Wp             = 0.25*ones(1,numel(NN.nodes)-1);
% Activation function for hidden layer {'tanh','sigm','cdf','relu','softplus'}
NN.hiddenLayerActivation = 'relu';        
% Activation function for hidden layer {'linear', 'tanh','sigm','cdf','relu'}
NN.outputActivation      = 'linear';  
% Optimize hyperparameters?
    % Observation noise sv 1: yes; 0: no
NN.optsv                 = 0;
    % Number of epochs 1: yes; 0: no 
NN.optNumEpochs          = 0;
    % Number of searching epochs
NN.numEpochs             = 0;
    % Number of Folds for cross-validation 
NN.numFolds              = 5;
    % Ratio between training set and validation set
NN.ratio                 = 0.8;
% Number of splits
NN.numSplits             = 20;
% Task regression or classification
NN.task                  = 'regression';
% Transfer data to GPU
if NN.gpu == 1
    x        = gpuArray(x);
    y        = gpuArray(y);
    trainIdx = gpuArray(trainIdx);
    testIdx  = gpuArray(testIdx);
end
%% Run
[mp, Sp, metric, time, hplist] = regression(NN, x, y, trainIdx, testIdx);
% Display results
disp('###################')
disp(' Final results')
disp(['  Avg. RMSE     : ' num2str(nanmean(metric.RMSElist)) ' +- ' num2str(nanstd(metric.RMSElist))])
disp(['  Avg. LL       : ' num2str(nanmean(metric.LLlist)) ' +- ' num2str(nanstd(metric.LLlist))])
disp(['  Avg. Time     : ' num2str(nanmean(time.trainTimelist)) ' +- ' num2str(nanstd(time.trainTimelist))])
disp(['  Avg. Opt time : ' num2str(nanmean(time.hpOptTimelist)) ' +- ' num2str(nanstd(time.hpOptTimelist))])
%% Save
folder   = char([cd ,'/results/']);
filename = 'power4layer_softplus_FV1.mat';
%save([folder filename],'NN', 'mp', 'Sp', 'metric', 'time', 'hplist')



