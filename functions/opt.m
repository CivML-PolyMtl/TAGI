%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         opt
% Description:  Optimize hyperparameters  for TAGI
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      December 9, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef opt
    methods (Static)
        function [bestmp, bestSp, Mvallist, optEpoch] = earlyStop(NN, mp, Sp, x, y)
            % Initializtion
            NNval           = NN;
            NNval.trainMode = 0;
            % Data
            trainIdx        = NN.trainIdx;
            valIdx          = NN.valIdx;
            if strcmp(NN.task, 'classification')
                [xtrain, ytrain, trainLabels, trainEncoderIdx] = dp.selectData(x, y, NN.labels, NN.encoderIdx, trainIdx);
                [xval, ~, valLabels, ~] = dp.selectData(x, y, NN.labels, NN.encoderIdx, valIdx);
                NN.encoderIdx = trainEncoderIdx;
                NN.labels     = trainLabels;
                NNval.labels  = valLabels;  
            elseif strcmp(NN.task, 'regression')
                [xtrain, ytrain, ~, ~] = dp.selectData(x, y, [], [], trainIdx);
                [xval, yval, ~, ~]     = dp.selectData(x, y, [], [], valIdx);
            end
            % Loop
            if NN.gpu == 1
                Mvallist = zeros(NN.numEpochs, 1, 'gpuArray');
            else
                Mvallist = zeros(NN.numEpochs, 1, NN.dtype);
            end
            stop     = 0;
            epoch    = 0;
            bestMval = 1;
            bestmp   = mp;
            bestSp   = Sp;
            optEpoch = 0;
            timeTot  = 0;
            while ~stop
                ts    = tic;
                epoch = epoch + 1;
                % Shuffle data
                if epoch >= 1
                    idxtrain      = randperm(size(ytrain, 1));
                    ytrain        = ytrain(idxtrain, :);
                    xtrain        = xtrain(idxtrain, :);
                    NN.encoderIdx = NN.encoderIdx(idxtrain, :);
                    NN.labels     = NN.labels(idxtrain);                    
                end 
                disp('#########################')
                disp(['Epoch #' num2str(epoch)])
                if strcmp(NN.task, 'classification')
                    NN.errorRateEval    = 0;
                    NNval.errorRateEval = 1;
                    [~, Mval, mp, Sp]   = opt.computeErrorRate(NN, NNval, mp, Sp,  xtrain, ytrain, xval); 
                elseif strcmp(NN.task, 'regression')
                    [~, Mval, mp, Sp, ~, ~] = opt.computeMetric(NN.sv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);  
                end
                Mvallist(epoch) = Mval;
                if Mval < bestMval
                    bestMval = Mval;
                    bestmp   = mp;
                    bestSp   = Sp;
                    optEpoch = epoch;
                end
                if epoch >= NN.numEpochs
                    stop = 1;
                end
                timeTot = timeTot + toc(ts);
                timeRem = timeTot/epoch*(NN.numEpochs-epoch)/60;
                disp(' ')
                disp(['Remaining time (mins): ' sprintf('%0.2f',timeRem)]);
            end
        end
        function [sv, N]             = crossValidation(NN, NNval, mp, Sp, x, y, optNumEpoch, numRuns)          
            % Split data into different folds
            numFolds = NN.numFolds;
            numEpochs= NN.numEpochs;
            numObs   = size(x, 1);
            initsv   = NN.sv;
            foldIdx  = dp.kfolds(numObs, numFolds);
            if strcmp(NN.task, 'classification')
                labels     = NN.labels;
                encoderIdx = NN.encoderIdx;
            end
            % Loop initialization
            if NN.gpu == 1
                 Nlist  = zeros(numRuns, 1, 'gpuArray');
                svlist  = zeros(numRuns, length(NN.sv), 'gpuArray');
            else
                Nlist  = zeros(numRuns, 1, NN.dtype);
                svlist = zeros(numRuns, length(NN.sv), NN.dtype);
            end                      
            dispFlag = 1;
            for n = 1:numRuns
                NN.sv    = initsv;
                if dispFlag == 1
                    disp('   ------')
                    disp(['   Fold #' num2str(n)])
                end
                [xtrain, xval]       = dp.regroup(x, foldIdx, n);
                [ytrain, yval]       = dp.regroup(y, foldIdx, n);
                if strcmp(NN.task, 'classification')
                    [trainLabels, valLabels] = dp.regroup(labels, foldIdx, n);
                    [trainEncoderIdx, valEncoderIdx] = dp.regroup(encoderIdx, foldIdx, n);
                    NN.labels        = trainLabels;
                    NNval.labels     = valLabels;
                    NN.encoderIdx    = trainEncoderIdx;
                    NNval.encoderIdx = valEncoderIdx;
                end
                [svlist(n, :), ~, ~] = opt.BGA(initsv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);
                if optNumEpoch == 1
                    Nlist(n)         = opt.numEpochs(numEpochs, svlist(n, :), NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);
                end
                if dispFlag == 1
                    fprintf(['   sigma_v      : ' repmat(['%#-+10.2e' ' '],[1, length(NN.sv)-1]) '%#-+10.2e\n'], svlist(n, :))
                    if optNumEpoch == 1
                        disp(['   Num. of epoch: ' num2str(Nlist(n))])
                    end
                end
            end
            sv = mean(svlist);
            N  = round(mean(Nlist));
        end
        function N                   = numEpochs(E, sv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval)
            if NN.gpu == 1
                Mvallist = zeros(E, 1, 'gpuArray');
            else
                Mvallist = zeros(E, 1, NN.dtype);
            end
            timeTot  = 0;
            for e = 1:E
                ts = tic;
                if e >= 1
                    idxtrain = randperm(size(ytrain, 1));
                    ytrain   = ytrain(idxtrain, :);
                    xtrain   = xtrain(idxtrain, :);
                    NN.encoderIdx = NN.encoderIdx(idxtrain, :);
                    NN.labels = NN.labels(idxtrain);                  
                end
                if strcmp(NN.task, 'classification')
                    disp('#########################')
                    disp(['Epoch #' num2str(e)])
                    NNval.errorRateEval = 1;
                    NN.errorRateEval    = 0;
                    [~, Mvallist(e), mp, Sp] = opt.computeErrorRate(NN, NNval, mp, Sp, xtrain, ytrain, xval);
                    if e == 1
                        % Save
                        nameNN     = [];
                        for n = 2:length(NN.nodes)-1
                            nameNN = [nameNN num2str(NN.nodes(n))];
                        end
                        filename = ['mnist' 'NN' nameNN  NN.hiddenLayerActivation  'SigmaV' num2str(NN.sv*10) 'ParamEpoch1'];
                        folder   = char([cd ,'/results/']);
                        save([folder filename], 'mp', 'Sp')
                    end
                    timeTot = timeTot + toc(ts);
                    timeRem = timeTot/e*(E-e)/60;
                    disp(' ')
                    disp(['Remaining time (mins): ' sprintf('%0.2f',timeRem)]);
                else
                    [~,  Mvallist(e), mp, Sp, ~, ~] = opt.computeMetric(sv, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);   
                end
            end
            [~, N] = max(Mvallist);        
        end
        function [thetaOpt, mp, Sp]  = BGA(theta, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval)
            % NR: Newton-Raphson; MMT: Momentum; ADAM: Adaptive moment
            % estimation
            optimizer           = 'ADAM'; 
            % Hyperparameters
            N                   = 50;
            displayFlag         = 0;
            learningRate        = 0.2;
            beta                = 0.9;
            beta1               = 0.9;
            beta2               = 0.999;
            epsilon             = 1E-8;
            tol                 = 1E-6;
            % Parameter setup
            numParams           = length(theta);
            idxParam            = 1*ones(numParams, 1);
            if NN.gpu == 1
                theta           = gpuArray(theta);
                thetaloop       = theta;
                thetaTR         = zeros(numParams, 1, 'gpuArray');
                momentumTR      = zeros(numParams, 1, 'gpuArray');
                vTR             = zeros(numParams, 1, 'gpuArray');
                sTR             = zeros(numParams, 1, 'gpuArray');
                gradientTR2OR   = zeros(numParams, 1, 'gpuArray');
                Mlist           = zeros(N+1, 1, 'gpuArray');
                Mvallist        = zeros(N+1, 1, 'gpuArray');
                thetalist       = zeros(N+1, numParams, 'gpuArray');
            else
                thetaloop       = theta;
                thetaTR         = zeros(numParams, 1);
                momentumTR      = zeros(numParams, 1);
                vTR             = zeros(numParams, 1);
                sTR             = zeros(numParams, 1);
                gradientTR2OR   = zeros(numParams, 1);
                Mlist           = zeros(N+1, 1);
                Mvallist        = zeros(N+1, 1);
                thetalist       = zeros(N+1, numParams);
            end
            funOR2TR            = cell(numParams, 1);
            funTR2OR            = cell(numParams, 1);
            funGradTR2OR        = cell(numParams, 1);
            for n = 1: numParams
                [funOR2TR{n}, funTR2OR{n}, funGradTR2OR{n}] = opt.getTransfFun(idxParam(n));
                thetaTR(n)  = funOR2TR{n}(theta(n));
            end
            % Compute inital metric
            NN.sv               = theta;
            NNval.sv            = theta;
            [M, Mval, mp, Sp, yf, Vf] = opt.computeMetric(theta, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);                      
            % Loop initialization
            converged           = 0;
            loop                = 0;
            count               = 0;
            evalGradient        = 1;           
            Mlist(1)            = M;
            Mvallist(1)         = Mval;
            thetalist(1, :)     = theta;
            if displayFlag == 1
                figure
%                 plot(Mlist(1), 'ok')
                hold on
                plot(Mvallist(1), 'om')
                xlabel('Num. of epoch', 'interpreter', 'latex')
                ylabel('Log-likelihood')
                xlim([1, N])
            end
            while ~converged
                loop     = loop + 1;
                if displayFlag == 1
                    disp(' ')
                    disp(['   Iteration #' num2str(loop)])
                end
                thetaRef = theta;
                % Compute gradient
                if evalGradient == 1
                    for n = 1:numParams
                        gradientTR2OR(n) = funGradTR2OR{n}(thetaTR(n));
                    end
                    if isfield(NN, 'encoderIdx')
                        ygrad = ytrain(NN.encoderIdx);
                    else
                        ygrad = ytrain;
                    end
                    [gradient, hessian]  = opt.computeGradient(ygrad, yf, Vf, theta);
                    gradientTR           = gradient.*gradientTR2OR;
                    hessianTR            = abs(hessian.*gradientTR2OR.^2);
                end
                % Update parameters
                if strcmp(optimizer, 'NR')
                    thetaTRloop                     = opt.NR(thetaTR, gradientTR, hessianTR);
                    momentumTRloop                  = nan;
                    vTRloop                         = nan;
                    sTRloop                         = nan;
                elseif strcmp(optimizer, 'MMT')
                    [thetaTRloop, momentumTRloop]   = opt.MMT(thetaTR, gradientTR, momentumTR, learningRate, beta);
                    vTRloop                         = nan;
                    sTRloop                         = nan;
                elseif strcmp(optimizer, 'ADAM')
                    [thetaTRloop, vTRloop, sTRloop] = opt.ADAM(thetaTR, sTR, vTR, gradientTR, learningRate, beta1, beta2, epsilon, loop);
                    momentumTRloop                  = nan;
                else
                    error ('The optimizer does not exist')
                end
                % Transform to original space
                for n = 1:numParams
                    thetaloop(n) = funTR2OR{n}(thetaTRloop(n));
                end
                % Compute metric w.r.t the new parameters
                [Mloop, Mvalloop, mploop, Sploop, yfloop, Vfloop] = opt.computeMetric(thetaloop, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval);
                % Update new parameter values for the next iteration 
                if Mloop > M
                    % Convergence check
                    if abs((M - Mloop) / M) < tol
                        converged = 1;
                    end
                    M           = Mloop;
                    Mval        = Mvalloop;
                    theta       = thetaloop;
                    thetaTR     = thetaTRloop;
                    mp          = mploop;
                    Sp          = Sploop;
                    momentumTR  = momentumTRloop;
                    vTR         = vTRloop;
                    sTR         = sTRloop;
                    yf          = yfloop;
                    Vf          = Vfloop;
                else
                    learningRate = learningRate/2; 
                    count        = count + 1;
                    evalGradient = 0;
                end
                % Savel to list
                Mlist(loop+1)         = M;
                Mvallist(loop+1)      = Mval;
                thetalist(loop+1, : ) = theta;
                % Convergence check
                if loop == N || count > 3 || converged == 1
                    [~, idx] = max(Mvallist(1:loop+1));  
                    thetaOpt = thetalist(idx, :);
                    break                   
                end
                % Display the results
                if displayFlag == 1
                    disp(['    Log likelihood: ' num2str(M)])
                    fprintf(['    current values: ' repmat(['%#-+15.2e' ' '],[1, numParams-1]) '%#-+15.2e\n',...
                        '      param change: ' repmat(['%#-+15.2e' ' '],[1, numParams-1]) '%#-+15.2e\n'], theta, theta-thetaRef)                   
%                     plot(loop+1, Mlist(loop+1), 'ok')
                    hold on
                    plot(loop+1, Mvallist(loop+1), 'om')
                    pause(0.01)
                end                
            end
        end      
        function [g, h]              = computeGradient(y, yf, Vf, sv)   
            d      = size(y, 2);
            sv     = sv.*ones(size(Vf));
            sigma  = Vf + sv.^2;
            B      = (y - yf).^2;
            if d == 1
                g  = mean(-sv./sigma + (sv./(sigma.^2)).*B);
                h  = mean(((sv.^2) - Vf)./(sigma.^2) + ((-3*(sv.^4) - 2*(sv.^2).*Vf + (Vf.^2))./(sigma.^4)).*B);
            else
                g  = mean(sum(d*(-sv)./sigma + (sv./(sigma.^2)).*B, 2));
                h  = mean(sum(((d*sv.^2) - Vf)./(sigma.^2) + ((-3*(sv.^4) - 2*(sv.^2).*Vf + (Vf.^2))./(sigma.^4)).*B, 2));
            end
        end
        function [M, Mval, mp, Sp, yf, Vf] = computeMetric(theta, NN, NNval, mp, Sp, xtrain, ytrain, xval, yval)
            NN.sv    = theta;
            NNval.sv = theta;
            % Training
            NN.trainMode            = 1;
            [mp, Sp, yf, Vf, ~, ~]     = tagi.network(NN, mp, Sp, xtrain, ytrain);
            if isfield(NN, 'encoderIdx')
                yf = yf(NN.encoderIdx);
                Vf = Vf(NN.encoderIdx);
                ytrain = ytrain(NN.encoderIdx);
            end
            Vf = Vf + (NN.sv.^2).*ones(size(Vf), NN.dtype);
            M  = mt.loglik(ytrain, yf, Vf);
            % Validation
            NNval.trainMode         = 0;
            [~, ~, yfval, Vfval, ~, ~] = tagi.network(NNval, mp, Sp, xval, []);
            if isfield(NNval, 'encoderIdx')
                yfval = yfval(NNval.encoderIdx);
                Vfval = Vfval(NNval.encoderIdx);
                yval  = yval(NNval.encoderIdx);
            end
            Vfval = Vfval + (NNval.sv.^2).*ones(size(Vfval));
            Mval  = mt.loglik(yval, yfval, Vfval);
        end
        function [M, Mval, mp, Sp]   = computeErrorRate(NN, NNval, mp, Sp,  xtrain, ytrain, xval)
            % Training
            NN.trainMode = 1;         
            [mp, Sp, ~, ~, ~, erTrain] = tagi.network(NN, mp, Sp, xtrain, ytrain);
            M = mean(erTrain);
            % Validation
            NNval.trainMode = 0;                   
            [~, ~, ~, ~, ~, erVal] = tagi.network(NNval, mp, Sp, xval, []);
            Mval = mean(erVal);
        end
        function theta               = NR(prevtheta, gradient, hessian)
            theta = prevtheta + gradient./hessian;
        end
        function [theta, momentum]   = MMT(prevtheta, gradient, prevMomentum, learningRate, beta)
            momentum = beta*prevMomentum + (1 - beta)*gradient;
            theta    = prevtheta + learningRate*momentum;
        end
        function [theta, v, s]       = ADAM(prevtheta, prevs, prevv, gradient, learningRate, beta1, beta2, epsilon, N)
            v       = beta2*prevv + (1 - beta2)*(gradient).^2;
            s       = beta1*prevs + (1 - beta1)*gradient;
            vhat    = v./(1-(beta2)^N);
            shat    = s./(1-(beta1)^N);
            theta   = prevtheta + learningRate*shat./(sqrt(vhat) + epsilon);
        end
        function [funOR2TR, funTR2OR, funGradTR2OR] = getTransfFun(idxParam)       
            if idxParam == 1  % loge
                transfOR2TR     = @(p) log(p);
                transfTR2OR     = @(p) exp(p);
                
                funOR2TR        = @(p) transfOR2TR(p);
                funTR2OR        = @(p) transfTR2OR(p);
                funGradTR2OR    = @(p) exp(p);                             
            elseif idxParam == 2  % log10
                transfOR2TR     = @(p) log10(p);
                transfTR2OR     = @(p) 10.^p;     
                
                funOR2TR        = @(p) transfOR2TR(p);
                funTR2OR        = @(p) transfTR2OR(p);
                funGradTR2OR    = @(p) log(10)*10.^p;                
            elseif idxParam == 3 % None
                transfOR2TR     = @(p) p;
                transfTR2OR     = @(p) p;
                
                funOR2TR        = @(p) transfOR2TR(p);
                funTR2OR        = @(p) transfTR2OR(p);
                funGradTR2OR    = @(p) 1;
            else
                error('Parameter transformation function are not properly defined in: config file')
            end
        end
    end
end