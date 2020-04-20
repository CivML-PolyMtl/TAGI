%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         dp
% Description:  Data processing
% Authors:      James-A. Goulet & Luong-Ha Nguyen
% Created:      November 8, 2019
% Updated:      January 23, 2020
% Contact:      james.goulet@polymtl.ca & luongha.nguyen@gmail.com 
% Copyright (c) 2020 James-A. Goulet & Luong-Ha nguyen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef dp
    methods (Static)
        % Data
        function [xntrain, yntrain, xntest, yntest, mxtrain, sxtrain, mytrain, sytrain] = normalize(xtrain, ytrain, xtest, ytest)
            mxtrain = nanmean(xtrain);
            sxtrain = nanstd(xtrain);
            idx     = sxtrain==0;
            sxtrain(idx) = 1;
            mytrain = nanmean(ytrain);
            sytrain = nanstd(ytrain);
            xntrain = (xtrain - mxtrain)./sxtrain;
%             xntrain(:,idx) = 0;
            yntrain = (ytrain - mytrain)./sytrain;
            xntest  = (xtest - mxtrain)./sxtrain;
%             xntest(:,idx) = 0;
            yntest  = ytest;
        end
        function [xtrain, ytrain, xtest, ytest] = split(x, y, ratio)
            numObs      = size(x, 1);
            %             idxobs      = 1:numObs;
            idxobs      = randperm(numObs);
            idxTrainEnd = round(ratio*numObs);
            idxTrain    = idxobs(1:idxTrainEnd);
            idxTest     = idxobs((idxTrainEnd+1):numObs);
            xtrain      = x(idxTrain, :);
            ytrain      = y(idxTrain, :);
            xtest       = x(idxTest, :);
            ytest       = y(idxTest, :);
        end
        function [trainIdx, testIdx] = indexSplit(numObs, ratio, dtype)
           idx = randperm(numObs);
           trainIdxEnd = round(numObs*ratio);
           trainIdx = idx(1:trainIdxEnd)';
           testIdx  = idx(trainIdxEnd+1:end)';
           if strcmp(dtype, 'single')
               trainIdx = int32(trainIdx);
               testIdx = int32(testIdx);
           end
        end
        function [x, y, labels, encoderIdx] = selectData(x, y, labels, encoderIdx, idx)
            x = x(idx, :);
            y = y(idx, :);
            if ~isempty(labels)
                labels = labels(idx, :);
            else
                labels = [];
            end
            if ~isempty(encoderIdx)
                encoderIdx = encoderIdx(idx, :);
            else
                encoderIdx = [];
            end
        end
        function foldIdx = kfolds(numObs, numFolds)
            numObsPerFold = round(numObs/(numFolds));
            idx           = 1:numObsPerFold:numObs;
            if ~ismember(numObs, idx)
                idx = [idx, numObs];
            end
            if length(idx)>numFolds+1
                idx(end-1) = []; 
            end
            foldIdx = cell(numFolds, 1);
            for i = 1:numFolds
                if i == numFolds
                    foldIdx{i} = [idx(i):idx(i+1)]';
                else
                    foldIdx{i} = [idx(i):idx(i+1)-1]';
                end
            end
        end       
        function [xtrain, xval] = regroup(x, foldIdx, valfold)
            trainfold       = 1:size(foldIdx, 1);
            trainfold(valfold) = [];
            xval            = x(foldIdx{valfold}, :);
            trainIdx        = cell2mat(foldIdx(trainfold));
            xtrain          = x(trainIdx, :);
        end
        function [y, sy] = denormalize(yn, syn, myntrain, syntrain)
            y   = yn.*syntrain + myntrain;
            if ~isempty(syn)
                sy  = (syntrain.^2).*syn;
            else
                sy  = [];
            end
        end
        function y  = transformObs(y)
            maxy    = 10;
            miny    = -10;
            idx     = logical(y);
            y(idx)  = maxy;
            y(~idx) = miny;
        end
        function prob  = probFromloglik(loglik)
            maxlogpdf = max(loglik);
            w_1       = bsxfun(@minus,loglik,maxlogpdf);
            w_2       = log(sum(exp(w_1)));
            w_3       = bsxfun(@minus,w_1,w_2);
            prob      = exp(w_3);
        end
        function [y, idx]   = encoder(yraw, numClasses, dtype)
            y   = zeros(size(yraw, 1), numClasses-1, dtype);
            if strcmp(dtype, 'single')
                idx = zeros(size(yraw, 1), 4, 'int32');
            elseif strcmp(dtype, 'double')
                idx = zeros(size(yraw, 1), 4, 'int64');
            end            
            for c = 1:numClasses
                idxClasses         = yraw==c-1;
                [idxLoop, obs]     = dp.class2obs(c, dtype, numClasses);
                y(idxClasses, idxLoop) = repmat(obs, [sum(idxClasses), 1]);
                idx(idxClasses, :) = repmat(idxLoop, [sum(idxClasses), 1]);
            end
        end
        function idx        = selectIndices(idx, batchSize, numClasses, dtype)
            if strcmp(dtype, 'single')
                numClasses = int32(numClasses);
            elseif strcmp(dtype, 'double')
                numClasses = int64(numClasses);
            end
            for b = 1:batchSize
                idx(b, :) = idx(b, :) + (b-1)*numClasses;
            end
            idx = reshape(idx', [size(idx, 1)*size(idx, 2), 1]);
        end
        function [obs, idx] = class_encoding(numClasses)
            H=ceil(log2(numClasses)); 
            C=fliplr(de2bi([0:numClasses-1],H));
            obs=(-1).^C;
            idx=ones(numClasses,H);
            C_sum=cumsum(1:H-1)+1;
            for i=1:numClasses
                for h=1:H-1
                    idx(i,h+1)=bi2de(fliplr(C(i,1:h)))+C_sum(h);
                end
            end
        end
        function [idx, obs] = class2obs(class, dtype, numClasses)
            [obs_c, idx_c]=dp.class_encoding(numClasses);
            idx=idx_c(class,:);
            obs=obs_c(class,:);
            if strcmp(dtype, 'single')
                idx = int32(idx);
                obs = single(obs);
            elseif strcmp(dtype, 'half')
                idx = int32(idx);
                obs = half(obs);
            end
        end
        function p_class    = obs2class(mz, Sz, dtype, numClasses)
            if strcmp(dtype,'half')
                mz=single(mz);
                Sz=single(Sz);
            end
            
            alpha = 3;
            p_obs = [normcdf(mz./sqrt((1/alpha)^2 + Sz), 0, 1); 1];
            [obs_c, idx_c]=dp.class_encoding(numClasses);
            p_class=prod(abs(p_obs(idx_c)-(obs_c==-1)),2);
        end
    end
end