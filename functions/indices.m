%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         indices
% Description:  Build indices for Bayesian linear neural networks
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 3, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef indices
    methods (Static)
        function NN = parameters(NN)
            % See document for the parameter's ordering
            % Initialization   
            if strcmp(NN.dtype, 'single')
                NN.nodes     = int32(NN.nodes);
                NN.batchSize = int32(NN.batchSize);
            end
            nodes     = NN.nodes;
            numLayers = length(nodes);           
            % Bias
            idxb      = cell(numLayers - 1, 1);
            % Weights
            idxw      = cell(numLayers - 1, 1);
            % Bias and weights
            idxbw     = cell(numLayers - 1, 1);             
            for j = 1:numLayers-1
                numParams  = nodes(j+1) + nodes(j+1)*nodes(j);
                idxbw{j}   = 1:numParams;
                idxb{j}    = idxbw{j}(1:nodes(j+1))';
                idxw{j}    = idxbw{j}(nodes(j+1)+1:numParams)';
            end
            NN.idxb  = idxb;
            NN.idxw  = idxw;
            NN.idxbw = idxbw;
        end
        function NN = covariance(NN)
            % Initialization
            batchSize       = NN.batchSize;
            numLayers       = length(NN.nodes);
            nodes           = NN.nodes;
            % Indices for F*mwa
            idxFmwa         = cell(numLayers - 1, 2);
            idxFmwab        = cell(numLayers - 1, 1);
            % Indices for F*Czwa
            idxFCzwa        = cell(numLayers - 1, 2);
            % Indices for activation unit
            idxa            = cell(numLayers - 1, 1);
            % Indices for F*Cwwa
            idxFCwwa        = cell(numLayers - 1, 2);
            % Indices for F*Cb
            idxFCb          = cell(numLayers - 1, 2);

            % Indices for updating parameters between layers
            idxSzpUd        = cell(numLayers - 1, 1);
            % Indices for updating hidden states between layers
            idxSzzUd        = cell(numLayers, 1);
            
            for j = 1:numLayers - 1   
                % Loop initialization
                dnext               = batchSize*nodes(j + 1);
                idxa{j}             = 1:NN.nodes(j)*batchSize;
                idxa{j}             = idxa{j}';
                idxaNext            = 1:nodes(j+1)*batchSize;
                % Get indices for F*mwa
                idxFmwa_1           = repmat(reshape(reshape(NN.idxw{j},[nodes(j + 1), nodes(j)])',[1, nodes(j + 1)*nodes(j)]), [1, batchSize]);
                idxFmwa_2           = reshape(repmat(reshape(idxa{j}, [nodes(j), batchSize]),[nodes(j + 1), 1]), [1, dnext*nodes(j)]);              
                if NN.gpu == 1
                    idxFmwa{j, 1}   = gpuArray(reshape(idxFmwa_1', [nodes(j), dnext])');
                    idxFmwa{j, 2}   = gpuArray(reshape(idxFmwa_2', [nodes(j), dnext])');
                    % Get indices for F*b
                    idxFmwab{j}     = gpuArray(repmat(NN.idxb{j}, [batchSize, 1]));
                    % Get indices for F*Cawa
                    if any(~isnan(NN.sx)) || j > 1
                        idxFCzwa{j, 1} = gpuArray(repmat(NN.idxw{j}, [batchSize, 1]));
                        idxFCzwa{j, 2} = gpuArray(reshape(repmat(idxa{j}', [nodes(j+1), 1]), [length(idxa{j})*nodes(j+1), 1]));                        
                    end
                    % Get indices for the parameter update step
                    idxSzpUd{j}     = gpuArray(repmat(reshape(idxaNext, [nodes(j + 1), batchSize]), [nodes(j)+1, 1]));
                    % Get indices for F*Cwwa
                        % Indices for Sp that uses to evaluate Cwwa
                    idxFCwwa{j, 1}  = gpuArray(reshape(repmat(NN.idxw{j}',[batchSize, 1]), [nodes(j)*dnext, 1]));
                        % Indices for ma that uses to evaluate Cwwa
                    idxFCwwa{j, 2}  = gpuArray(reshape(repmat(reshape(idxa{j}, [nodes(j), batchSize])',[nodes(j + 1), 1]),[nodes(j)*dnext, 1]));
                    % Get indices for F*Sb
                    idxFCb{j}       = gpuArray(reshape(repmat(NN.idxb{j}', [batchSize, 1]), [dnext,1]));
                else
                    % Indices on CPU
                    idxFmwa{j, 1}   = reshape(idxFmwa_1', [nodes(j), dnext])';
                    idxFmwa{j, 2}   = reshape(idxFmwa_2', [nodes(j), dnext])';
                    idxFmwab{j}     = repmat(NN.idxb{j}, [batchSize, 1]);
                    if any(~isnan(NN.sx)) || j > 1
                        idxFCzwa{j, 1} = repmat(NN.idxw{j}, [batchSize, 1]);
                        idxFCzwa{j, 2} = reshape(repmat(idxa{j}', [nodes(j+1), 1]), [length(idxa{j})*nodes(j+1), 1]);
                    end
                    idxSzpUd{j}     = repmat(reshape(idxaNext, [nodes(j + 1), batchSize]), [nodes(j)+1, 1]);
                    idxFCwwa{j, 1}  = reshape(repmat(NN.idxw{j}',[batchSize, 1]), [nodes(j)*dnext, 1]);
                    idxFCwwa{j, 2}  = reshape(repmat(reshape(idxa{j}, [nodes(j), batchSize])',[nodes(j + 1), 1]),[nodes(j)*dnext, 1]);
                    idxFCb{j}       = reshape(repmat(NN.idxb{j}', [batchSize, 1]), [dnext,1]);
                end
                % Get indices for the hidden state update step
                if NN.gpu == 1
                    idxSzzUd{j} = gpuArray(reshape(repmat(reshape(idxaNext, [nodes(j + 1), batchSize]), [nodes(j), 1]), [nodes(j+1), nodes(j)*batchSize])');
                else
                    idxSzzUd{j} = reshape(repmat(reshape(idxaNext, [nodes(j + 1), batchSize]), [nodes(j), 1]), [nodes(j+1), nodes(j)*batchSize])';
                end                               
            end
            % Outputs
                % F*mwa
            NN.idxFmwa      = idxFmwa;
            NN.idxFmwab     = idxFmwab;
                % F*Cawa
            NN.idxFCzwa     = idxFCzwa;
            NN.idxSzpUd     = idxSzpUd;
                % Caa
            NN.idxSzzUd     = idxSzzUd;
                % F*Cwwa
            NN.idxFCwwa     = idxFCwwa;               
                % a
            NN.idxa         = idxa;  
            NN.idxFCb       = idxFCb;
        end
    end
end