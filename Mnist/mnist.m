%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         mnist
% Description:  Apply TAGI to mnist 
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      December 19, 2019
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
path                      = char([cd ,'/data/']);
load(char([path, '/mnistTrain.mat']))
x                         = x_obs;
y                         = y_obs;
path                      = char([cd ,'/data/']);
load(char([path, '/mnistTest.mat']))
x                         = [x; x_obs];
y                         = [y; y_obs];
Nclass                    = 10;
idxN                      = y < Nclass;
trainIdx                  = 1:60000;
testIdx                   = 60001:70000;
x                         = single(x)/255;
y                         = single(y(idxN));
yref                      = y;
x                         = x(idxN,:);
nx                        = size(x, 2);  
%% Neural Network properties
% GPU 1: yes; 0: no
NN.gpu                   = 0;
% Data type object half or double precision
NN.dtype                 = 'single';
% Encoding
[y , encoderIdx]          = dp.encoder(y, Nclass, NN.dtype);      
ny                        = size(y, 2); 
% Number of input covariates
NN.nx                    = nx; 
% Number of output responses
NN.ny                    = ny;   
% Number of classes
NN.numClasses            = Nclass;
% Batch size [train, val, test]
NN.batchSizelist         = [1]; 
% Number of nodes for each layer
NN.nodes                 = [NN.nx 100 100 NN.ny]; 
% Input standard deviation        
NN.sx                    = nan;
% Observations standard deviation
NN.sv                    = 0.2*ones(1,1);        
% Maximal number of learnign epoch
NN.maxEpoch              = 1;   
% Factor for initializing weights & bias
NN.factor4Bp             = 1E-2*ones(1,numel(NN.nodes)-1);
NN.factor4Wp             = 1*ones(1,numel(NN.nodes)-1);
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
NN.numEpochs             = 40;
    % Number of Folds for cross-validation 
NN.numFolds              = 0;
% Babysittig learning process for evaluating the model's performance
NN.earlyStop             = 0;
    % Ratio between training set and validation set
NN.ratio                 = 0.9;
% Encoder indices for the classification task
NN.encoderIdx            = encoderIdx;
% Labels for the observation
NN.labels                = yref;
% Task
NN.task                  = 'classification';
% Transfer data to GPU
if NN.gpu == 1
    x        = gpuArray(x);
    y        = gpuArray(y);
    trainIdx = gpuArray(trainIdx);
    testIdx  = gpuArray(testIdx);
end

% Parameters
NN.trainMode = 1;  
NN.batchSize = NN.batchSizelist(1);
NN           = indices.parameters(NN);
NN           = indices.covariance(NN);
NN.cd        = char([cd ,'/data/']);
param        = load([NN.cd, 'mnistNN100100reluEarlyStopInitParam_1.mat']);
mp           = param.mp;
Sp           = param.Sp;
if NN.gpu == 1
    mp{1}   = gpuArray(mp{1});
    mp{2}   = gpuArray(mp{2});
    mp{3}   = gpuArray(mp{3});
    Sp{1}   = gpuArray(Sp{1});
    Sp{2}   = gpuArray(Sp{2});
    Sp{3}   = gpuArray(Sp{3});
end
idx         = load([NN.cd 'mnistTrainIdx_1.mat']);
NN.trainIdx = idx.trainIdx;
NN.valIdx   = idx.valIdx;
%% Run
[mp, Sp, metric, time, hp, testIdx] = classification(NN, x, y, mp, Sp, trainIdx, testIdx);
NN.sv       = hp(1);
NN.maxEpoch = hp(2);
disp(' ')
disp(' Final results ')
disp([' Error rate    : ' num2str(mean(metric.erTest))])
disp([' training time : ' num2str(time.trainTimelist/60)])
disp([' tuning time   : ' num2str(time.hpOptTimelist/60)])
%% Save
 nameNN     = [];
for n = 2:length(NN.nodes)-1
    nameNN = [nameNN num2str(NN.nodes(n))];
end
filename = ['mnist' 'NN' nameNN  NN.hiddenLayerActivation 'B' num2str(NN.batchSizelist(1)) 'Epoch' num2str(hp(2)) 'SigmaV' num2str(NN.sv*10) 'EarlyStop' 'run1'];
folder   = char([cd ,'/results/']);
%save([folder filename],'NN', 'mp', 'Sp', 'metric', 'time', 'hp', 'testIdx')

%% Plot class probabilities for the test set
y_test=y_obs;
pr=sortrows([metric.PnTest(:,1:10),y_test],11);
idx_y=0;
for i=0:9
    idx=find(pr(:,11)==i);
    pr(idx,:)=sortrows(pr(idx,:),-(i+1));
    idx_y=[idx_y,idx_y(end)+numel(idx)];
end

p_class=[0.09:0.01:0.99 0.995 0.997 0.998 0.999];
class_summary=zeros(10000,length(p_class));
for i=1:10000
    for j=1:length(p_class)
        if and(pr(i,pr(i,end)+1)==max(pr(i,1:10)),pr(i,pr(i,end)+1)>=p_class(j))
            class_summary(i,j)=1;
        elseif all(pr(i,1:10)<p_class(j))
            class_summary(i,j)=2;
        elseif and(pr(i,pr(i,end)+1)~=max(pr(i,1:10)),max(pr(i,1:10))>=p_class(j))
            class_summary(i,j)=3;
        end
    end
end
stat=[mean(class_summary==1);mean(class_summary==2);mean(class_summary==3)];


figure('Position', [0 0 1000 300]);
subplot(1,3,1:2)
h=imagesc(pr(:,1:10)',[0,1])

colormap(jet)
colorbar
ax = gca;
ax.XTick = idx_y;
ax.YTick = [-0.5:1:10.5];
ax.GridColor = [1 1 1];
ax.GridAlpha = 1;
xlim([0,10000])
xticklabels({'  0','  1','  2','  3','  4','  5','  6','  7','  8','  9',''})
yticklabels({'','0','','2','','4','','6','','8','',''})
ylabel(['$\Pr($' 'labels' '$|\mathcal{D})$'],'Interpreter','Latex')
xlabel('Test set labels (0-9)')
grid on



subplot(1,3,3)
h=area(p_class,[stat]');
h(1).FaceColor = 'white';
h(2).FaceColor = [0.5 0.5 0.5];
h(3).FaceColor = 'black';
xlim([0.1,0.999]);
ax = gca;
ax.XTick = [0.1 0.5 0.9];

xlabel('$\phi$','Interpreter','Latex')
ylabel('$\Pr(~)$','Interpreter','Latex')
legend({'Pr(correct class)','Pr(unknown)','Pr(incorrect class)' })

