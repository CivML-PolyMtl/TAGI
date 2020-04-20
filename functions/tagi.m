%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         tagi
% Description:  tractable approximate gaussian inference 
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 3, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef tagi
    methods(Static) 
        % Main network
        function [mp, Sp, zn, Szn, Pn, er]   = network(NN, mp, Sp, x, y)
            % Initialization
            numObs        = size(x, 1);
            numCovariates = NN.nx;
            if ~isfield(NN, 'errorRateEval')
                NN.errorRateMonitoring = 0;
            end
            if NN.gpu == 1
                zn  = zeros(numObs, NN.ny, 'gpuArray');
                Szn = zeros(numObs, NN.ny, 'gpuArray');               
            else
                zn  = zeros(numObs, NN.ny, NN.dtype);
                Szn = zeros(numObs, NN.ny, NN.dtype);
            end
            if NN.errorRateEval == 1
                if NN.gpu == 1
                    Pn = zeros(numObs, NN.numClasses, 'gpuArray');
                    P  = zeros(NN.batchSize, NN.numClasses, 'gpuArray');
                    er = zeros(numObs, 1, 'gpuArray');
                else
                    Pn = zeros(numObs, NN.numClasses);
                    P  = zeros(NN.batchSize, NN.numClasses, NN.dtype);
                    er = zeros(numObs, 1, NN.dtype);
                end
            else
                er = nan;
                Pn = nan;
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:NN.batchSize:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i:i+NN.batchSize-1;
                xloop    = reshape(x(idxBatch, :)', [length(idxBatch)*numCovariates, 1]);   
                % Training
                if NN.trainMode == 1
                    yloop            = reshape(y(idxBatch, :)', [length(idxBatch)*NN.ny, 1]);
                    if isfield(NN, 'encoderIdx')
                        updateIdx    = dp.selectIndices(NN.encoderIdx(idxBatch, :), NN.batchSize, NN.ny, NN.dtype);
                    else
                        updateIdx    = [];
                    end
                    [mz, Sz, Czw, Czb, Czz] = tagi.feedForward(NN, xloop, mp, Sp);
                    [mp, Sp]         = tagi.feedBackward(NN, mp, Sp, mz, Sz, Czw, Czb, Czz, yloop, updateIdx);
                    zn(idxBatch, :)  = reshape(mz{end}, [NN.ny, length(idxBatch)])';
                    Szn(idxBatch, :) = reshape(Sz{end}, [NN.ny, length(idxBatch)])';
                % Testing    
                else 
                    [mz, Sz, ~, ~]   = tagi.feedForward(NN, xloop, mp, Sp);
                    zn(idxBatch, :)  = reshape(mz, [NN.ny, length(idxBatch)])';
                    Szn(idxBatch, :) = reshape(Sz, [NN.ny, length(idxBatch)])';
                end
                % Error rate
                if NN.errorRateEval == 1
                    zi              = zn(idxBatch, :);
                    Szi             = Szn(idxBatch, :);
                    for j = 1:NN.batchSize
                        P(j,:)      = dp.obs2class(zi(j,:)', Szi(j,:)' + NN.sv.^2, NN.dtype, NN.numClasses);
                        P(j,:)   = P(j,:)/sum(P(j,:));
                    end
                    Pn(idxBatch, :) = P;
                    er(idxBatch, :) = mt.errorRate(NN.labels(idxBatch, :)', P');
                    time_loop       = toc(timeval);
                    time_tot        = time_tot + time_loop;
                    time_rem        = double(time_tot)/(double(idxBatch(end)))*(numObs-double(idxBatch(end)))/60;
                    % Display error rate  
                    if mod(idxBatch(end), NN.obsShow) == 0 && i > 1 && NN.trainMode == 1
                        disp(['  Error Rate (%)  : ' sprintf('%0.3f', mean(er(max(1,i-100):i)))]);
                        disp(['  Time left (mins): ' sprintf('%0.2f',time_rem)])
                    end
                end                               
            end
        end
        % Foward network
        function [mz, Sz, Czw, Czb, Czz] = feedForward(NN, x, mp, Sp)
            % Initialization
            gpu                  = NN.gpu;
            numLayers            = length(NN.nodes);
            hiddenLayerActFunIdx = act.activationFunIndex(NN.hiddenLayerActivation); 
            % Activation unit
            ma                   = cell(numLayers, 1); 
            ma{1}                = x;  
            Sa                   = cell(numLayers, 1); 
            % Hidden states
            mz                   = cell(numLayers, 1); 
            Czw                  = cell(numLayers, 1);
            Czb                  = cell(numLayers, 1);
            Czz                  = cell(numLayers, 1); 
            Sz                   = cell(numLayers, 1); 
            J                    = cell(numLayers, 1);           
            % Hidden Layers
            for j = 2:numLayers
                mz{j} = tagi.meanMz(mp{j-1}, ma{j-1}, NN.idxFmwa(j-1, :), NN.idxFmwab{j-1}, gpu);
                % Covariance for z^(j)
                Sz{j} = tagi.covarianceSz(mp{j-1}, ma{j-1}, Sp{j-1}, Sa{j-1}, NN.idxFmwa(j-1, :), NN.idxFmwab{j-1}, gpu);
                if NN.trainMode == 1
                    % Covariance between z^(j) and w^(j-1) 
                    [Czw{j}, Czb{j}] = tagi.covarianceCzp(ma{j-1}, Sp{j-1}, NN.idxFCwwa(j-1, :), NN.idxFCb{j-1}, gpu); 
                    % Covariance between z^(j+1) and z^(j)
                    if ~isempty(Sz{j-1})
                        Czz{j} = tagi.covarianceCzz(mp{j-1}, Sz{j-1}, J{j-1}, NN.idxFCzwa(j-1, :), gpu);
                    end
                end
                % Activation
                if j < numLayers
                    [ma{j}, J{j}]  = act.meanA(mz{j}, mz{j}, hiddenLayerActFunIdx, gpu); 
                    Sa{j}          = act.covarianceSa(J{j}, Sz{j}, gpu);
                end               
            end
            % Outputs
            if ~strcmp(NN.outputActivation, 'linear')
                ouputLayerActFunIdx = act.activationFunIndex(NN.outputActivation);
                [mz{numLayers}, J]  = act.meanA(mz{numLayers}, mz{numLayers}, ouputLayerActFunIdx, gpu);
                Sz{numLayers}       = act.covarianceSa(J, Sz{numLayers}, gpu);               
            end 
            if NN.trainMode == 0
                mz  = mz{numLayers};
                Sz  = Sz{numLayers};
            end            
        end
        % Backward network
        function [mpUd, SpUd] = feedBackward(NN, mp, Sp, mz, Sz, Czw, Czb, Czz, y, udIdx)
            % Initialization
            numLayers = length(NN.nodes);
            mpUd      = cell(numLayers - 1, 1);
            SpUd      = cell(numLayers - 1, 1);
            mzUd      = cell(numLayers - 1, 1);
            SzUd      = cell(numLayers - 1, 1);
            lHL       = numLayers-1; 
            if NN.ny == length(NN.sv)
                sv    = NN.sv';
            else
                sv    = repmat(NN.sv, [NN.ny, 1]);
            end
            % Update hidden states for the last hidden layer 
            R                          = repmat(sv.^2, [NN.batchSize, 1]);
            Szv                        = Sz{lHL+1} + R;   
            if isempty(udIdx)
                [mzUd{lHL+1}, SzUd{lHL+1}] = tagi.fowardHiddenStateUpdate(mz{lHL+1}, Sz{lHL+1}, mz{lHL+1}, Szv, Sz{lHL+1}, y, NN.gpu);
            else
                mzf                    = mz{lHL+1}(udIdx);
                Szf                    = Sz{lHL+1}(udIdx);
                ys                     = y(udIdx);
                Szv                    = Szv(udIdx);
                mzUd{lHL+1}            = mz{lHL+1};
                SzUd{lHL+1}            = Sz{lHL+1};
                [mzUd{lHL+1}(udIdx), SzUd{lHL+1}(udIdx)] = tagi.fowardHiddenStateUpdate(mzf, Szf, mzf, Szv, Szf, ys, NN.gpu);
            end          
            for k = (numLayers-1):-1:1  
                % Update parameters
                Czp                    = tagi.buildCzp(Czw{k+1}, Czb{k+1}, NN.nodes(k+1), NN.nodes(k), NN.batchSize);
                [mpUd{k}, SpUd{k}]     = tagi.backwardParameterUpdate(mp{k}, Sp{k}, mz{k+1}, Sz{k+1}, SzUd{k+1}, Czp, mzUd{k+1}, NN.idxSzpUd{k}, NN.gpu); 
                % Update hidden states
                if k > 1
                     Czzloop           = tagi.buildCzz(Czz{k+1}, NN.nodes(k+1), NN.nodes(k), NN.batchSize);
                    [mzUd{k}, SzUd{k}] = tagi.backwardHiddeStateUpdate(mz{k}, Sz{k}, mz{k+1}, Sz{k+1}, SzUd{k+1}, Czzloop, mzUd{k+1}, NN.idxSzzUd{k}, NN.gpu);
                end
            end
        end
        % Covariance
        function mz  = meanMz(mp, ma, idxFmwa, idxFmwab, gpu)
            % mp is the mean of parameters for the current layer
            % ma is the mean of activation unit (a) from previous layer           
            % Cpa is covariance matrix between parameters p and a
            % idxFmwa{1} is the indices for weight w
            % idxFmwa{2} is the indices for activation unit a
            % idxFmwab is the indices for bias b
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % *NOTE*: All indices have been built in the way that we bypass
            % the transition step such as F*mwa + F*b
            if size(idxFmwa{1}, 1) == 1
                idxSum = 1;
            else
                idxSum = 2;
            end
            mpb = mp(idxFmwab);
            mp  = mp(idxFmwa{1});
            ma  = ma(idxFmwa{2});
            if gpu == 1
                mz  = bsxfun(@plus, sum(bsxfun(@times, mp, ma), idxSum), mpb);                            
            else
                mWa = sum(mp.*ma, idxSum);
                mz  = mWa + mpb;
            end            
        end
        function Sz  = covarianceSz(mp, ma, Sp, Sa, idxFSwaF, idxFSwaFb, gpu)
            % mp is the mean of parameters for the current layer
            % ma is the mean of activation unit (a) from previous layer           
            % Sp is the covariance matrix for parameters p
            % Sa is the covariance matrix for a from the previous layer
            % idxFSwaF{1} is the indices for weight w
            % idxFSwaF{2} is the indices for activation unit a
            % idxFSwaFb is the indices for bias
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % *NOTE*: All indices have been built in the way that we bypass
            % the transition step such as Sa = F*Cwa*F' + F*Cb*F'
            if size(idxFSwaF{1}, 1) == 1
                idxSum = 1;
            else
                idxSum = 2;
            end
            Spb = Sp(idxFSwaFb);           
            ma  = ma(idxFSwaF{2});
            Sp  = Sp(idxFSwaF{1});           
            if gpu == 1
                if isempty(Sa)
                    Sa1   = bsxfun(@times, Sp, ma);
                    Sz    = sum(bsxfun(@times, Sa1, ma), idxSum);
                else
                    mp    = mp(idxFSwaF{1});
                    Sa    = Sa(idxFSwaF{2});
                    Sa1   = bsxfun(@times, Sp, Sa);
                    Sa2   = bsxfun(@times, Sp, ma); 
                    Sa2   = bsxfun(@times, Sa2, ma); 
                    Sa3   = bsxfun(@times, Sa, mp); 
                    Sa3   = bsxfun(@times, Sa3, mp); 
                    Sa12  = bsxfun(@plus, Sa1, Sa2);
                    Sa123 = bsxfun(@plus, Sa12, Sa3);
                    Sz    = sum(Sa123, idxSum);                    
                end
                Sz  = bsxfun(@plus, Sz, Spb);
            else
                if isempty(Sa)
                    Sz  = sum(Sp.*ma.*ma, idxSum);
                else
                    mp  = mp(idxFSwaF{1});
                    Sa  = Sa(idxFSwaF{2});
                    Sz  = sum(Sp.*Sa + Sp.*ma.*ma + Sa.*mp.*mp, idxSum);
                end
                Sz = Sz + Spb;
            end
        end   
        function [Czw, Czb] = covarianceCzp(ma, Sp, idxFCwwa, idxFCb, gpu)
            % ma is the mean of activation unit (a) from previous layer           
            % Sp is the covariance matrix for parameters p
            % idxFCpwa{1} is the indices for weight w
            % idxFCpwa{2} is the indices for weight action unit a
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % *NOTE*: All indices have been built in the way that we bypass
            % the transition step such as Cpa = F*Cpwa + F*Cb
            
            Czb = Sp(idxFCb);   
            Sp  = Sp(idxFCwwa{1});
            ma  = ma(idxFCwwa{2});
            if gpu == 1
                Czw = bsxfun(@times, Sp, ma);             
            else
                Czw = Sp.*ma;
            end                    
        end 
        function Czz = covarianceCzz(mp, Sz, J, idxCawa, gpu)
            Sz = Sz(idxCawa{2});
            mp = mp(idxCawa{1});
            J  = J(idxCawa{2});
            if gpu == 1
                Czz = bsxfun(@times, J, Sz);
                Czz = bsxfun(@times, Czz, mp);
            else
                Czz = J.*Sz.*mp;
            end
        end
        % Update Step
        function [mpUd, SpUd] = backwardParameterUpdate(mp, Sp, mzF, SzF, SzB, Czp, mzB, idx, gpu)
            if gpu == 1
                dz   = bsxfun(@minus, mzB, mzF);
                dz   = dz(idx);
                dS   = bsxfun(@minus, SzB, SzF);
                dS   = dS(idx);
                SzF  = bsxfun(@rdivide, 1, SzF);
                SzF  = SzF(idx);
                J    = bsxfun(@times, Czp, SzF);               
                % Mean
                mpUd = sum(bsxfun(@times, J, dz), 2);
                mpUd = bsxfun(@plus, mp, mpUd);
                % Covariance
                SpUd = bsxfun(@times, J, dS);
                SpUd = sum(bsxfun(@times, SpUd, J), 2);
                SpUd = bsxfun(@plus, Sp, SpUd);
            else
                dz   = mzB - mzF;
                dz   = dz(idx);
                dS   = SzB - SzF;
                dS   = dS(idx);
                SzF  = 1./SzF;
                SzF  = SzF(idx);
                J    = Czp.*SzF;
                % Mean
                mpUd = mp + sum(J.*dz, 2);
                % Covariance
                SpUd = Sp + sum(J.*dS.*J, 2);
            end
        end
        function [mzUd, SzUd] = backwardHiddeStateUpdate(mz, Sz, mzF, SzF, SzB, Czz, mzB, idx, gpu)
            if gpu == 1
                dz   = bsxfun(@minus, mzB, mzF);
                dz   = dz(idx);
                dS   = bsxfun(@minus, SzB, SzF);
                dS   = dS(idx);
                SzF  = bsxfun(@rdivide, 1, SzF);
                SzF  = SzF(idx);
                J    = bsxfun(@times, Czz, SzF);   
                % Mean
                mzUd = sum(bsxfun(@times, J, dz), 2);
                mzUd = bsxfun(@plus, mz, mzUd);
                % Covariance
                SzUd = bsxfun(@times, J, dS); 
                SzUd = sum(bsxfun(@times, SzUd, J), 2);
                SzUd = bsxfun(@plus, Sz, SzUd);
            else
                dz   = mzB - mzF;
                dz   = dz(idx);
                dS   = SzB - SzF;
                dS   = dS(idx);
                SzF  = 1./SzF;
                SzF  = SzF(idx);
                J    = Czz.*SzF;               
                % Mean
                mzUd = mz + sum(J.*dz, 2);
                % Covariance
                SzUd = Sz + sum(J.*dS.*J, 2);
            end
        end
        function [mzUd, SzUd] = fowardHiddenStateUpdate(mz, Sz, mzF, SzF, Cyz, y, gpu)
            if gpu == 1
                dz   = bsxfun(@minus, y, mzF);
                SzF  = bsxfun(@rdivide, 1, SzF);
                K    = bsxfun(@times, Cyz, SzF);
                % Mean
                mzUd = bsxfun(@times, K, dz);
                mzUd = bsxfun(@plus, mz, mzUd);
                % Covariance
                SzUd = bsxfun(@times, K, Cyz);
                SzUd = bsxfun(@minus, Sz, SzUd);
            else
                dz   = y - mzF;
                SzF  = 1./SzF;
                K    = Cyz.*SzF;
                % Mean
                mzUd = mz + K.*dz;
                % Covariance
                SzUd = Sz - K.*Cyz;
            end
        end
        % Build the matrix Czp, Czz for the update step
        function Czp = buildCzp(Czw, Czb, currentHiddenUnit, prevHiddenUnit, batchSize)
            Czp = [Czb; Czw];
            Czp = reshape(Czp, [batchSize, currentHiddenUnit*prevHiddenUnit+currentHiddenUnit])';
        end
        function Czz = buildCzz(Czz, currentHiddenUnit, prevHiddenUnit, batchSize)
            Czz = reshape(Czz, [currentHiddenUnit, prevHiddenUnit*batchSize])';
        end
        % Initialization weights and bias
        function [mp, Sp] = initializeWeightBias(NN)
            % Initialization
            NN.dropWeight  = 0;
            nodes          = double(NN.nodes);
            numLayers      = length(NN.nodes);
            idxw           = NN.idxw;
            idxb           = NN.idxb;
            if strcmp(NN.dtype , 'single')
                factor4Bp  = single(NN.factor4Bp);
                factor4Wp  = single(NN.factor4Wp); 
                nodes      = single(nodes);
            else
                factor4Bp  = double(NN.factor4Bp);
                factor4Wp  = double(NN.factor4Wp); 
                nodes      = double(nodes);
            end             
            mp         = cell(numLayers - 1, 1);
            Sp         = cell(numLayers - 1, 1);
            for j = 2:numLayers
                % Bias variance
                Sbwloop{1}     = factor4Bp(j-1)*ones(1, length(idxb{j-1}), NN.dtype);
                % Bias mean
                bwloop{1}      = randn(1, length(Sbwloop{1})).*sqrt(Sbwloop{1});
                % Weight variance
                if strcmp(NN.hiddenLayerActivation, 'relu') || strcmp(NN.hiddenLayerActivation, 'softplus') || strcmp(NN.hiddenLayerActivation, 'sigm')
                    Sbwloop{2} = factor4Wp(j-1)*(1/(nodes(j-1)))*ones(1, length(idxw{j-1}), NN.dtype);
                else
                    Sbwloop{2} = factor4Wp(j-1)*(2/(nodes(j-1) + nodes(j)))*ones(1, length(idxw{j-1}), NN.dtype);
                end
                % Weight mean
                bwloop{2}      = randn(1, length(Sbwloop{2})).*sqrt(Sbwloop{2});
                %Set weight to zeros
                idxwset_0      = randperm(length(idxw{j-1}),round(NN.dropWeight*length(idxw{j-1})));               
                bwloop{2}(idxwset_0)= 0;
                % Save
                if NN.gpu ==1
                    mp{j-1}    = gpuArray(cell2mat(bwloop)');
                    Sp{j-1}    = gpuArray(cell2mat(Sbwloop)');
                else
                    mp{j-1}    = cell2mat(bwloop)';
                    Sp{j-1}    = cell2mat(Sbwloop)';
                end
            end 
        end        
    end
end