%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         act
% Description:  Activation function
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 12, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef act   
    methods (Static)
        function funIdx = activationFunIndex(funName)
            if strcmp(funName,'tanh')
                funIdx = 1;
            elseif strcmp(funName,'sigm')
                funIdx = 2;
            elseif strcmp(funName,'cdf')
                funIdx = 3;
            elseif strcmp(funName,'relu')
                funIdx = 4;
            elseif strcmp(funName, 'softplus')
                funIdx = 5;
            end
        end
        function [s, J] = meanA(z, mz, funIdx, gpu)         
            if funIdx == 1 % tanh
                if gpu
                    tanh_mz     = tanh(mz);
                    dtanh_mz    = bsxfun(@power,tanh_mz,2);
                    dtanh_mz    = bsxfun(@minus,1,dtanh_mz);
                    
                    tanh_z      = tanh(z);
                    dtanh_z     = bsxfun(@power, tanh_z,2);
                    dtanh_z     = bsxfun(@minus,1,dtanh_z);
                    
                    s           = bsxfun(@minus,z,mz);
                    s           = bsxfun(@times,dtanh_mz,s);
                    s           = bsxfun(@plus,s,tanh_mz);
                      
                    J           =  dtanh_z; 
                else
                    dtanhf      = @(x) 1-tanh(x).^2;
                    s           = dtanhf(mz).*(z-mz)+tanh(mz);
                    J           = dtanhf(z);  
                end
            elseif funIdx == 2 % sigmoid
                if gpu
                    sigmoid_mz  = exp(-mz);
                    sigmoid_mz  = bsxfun(@plus, 1, sigmoid_mz);
                    sigmoid_mz  = bsxfun(@rdivide, 1, sigmoid_mz);
                    
                    dsigmoid_mz = bsxfun(@minus, 1, sigmoid_mz);
                    dsigmoid_mz = bsxfun(@times, sigmoid_mz, dsigmoid_mz);                   
                    
                    s           = sigmoid_mz;                     
                    J           = dsigmoid_mz;  
                else
                    sigmoid     = @(x) 1./(1+exp(-x));
                    dsigmoid    = @(x) sigmoid(x).*(1-sigmoid(x));
                    s           = sigmoid(mz);
                    J           = dsigmoid(z);
                end
            elseif funIdx == 3 % cdf
                if gpu
                    npdf_mz     = normpdf(mz);
                    ncdf_mz     = normcdf(mz);
                    npdf_z      = normpdf(z);
                    
                    s           = bsxfun(@minus,z,mz);
                    s           = bsxfun(@times,npdf_mz,s);
                    s           = bsxfun(@plus,s,ncdf_mz);
                                            
                    J           =  npdf_z;
                else
                    s           = normpdf(mz).*(z-mz)+normcdf(mz);
                    J           = normpdf(z);
                end
            elseif funIdx == 4 % relu
                if gpu
                    max_mz      = mz;
                    idx         = mz<0;
                    max_mz(idx) = 0;
                    J           = single(~idx);                  
                    s           = max_mz;                            
                else
                    s           = max(0, mz);
                    J           = single(z>0);
                end            
            elseif funIdx == 5 % softplus
                if gpu
                    alpha=1;
                    k = alpha*mz<30;
                    e = bsxfun(@plus, 1, exp(alpha*mz.*k));
                    s = (log(e) + mz.*(1-k))/alpha;
                    J = k.*bsxfun(@rdivide, exp(alpha*mz.*k), e) + (1-k);
                else
                    alpha=10;
                    k = alpha*mz<30;
                    s = 1 + exp(alpha*mz.*k);
                    s = (log(s) + mz.*(1-k))/alpha;
                    J = k.*exp(alpha*mz.*k)./(1 + exp(alpha*mz.*k)) + (1-k)/alpha;
                end
            end        
        end
        function Cwa    = covarianceCwa(J, Cwz, idxJ, gpu)
            if gpu
                J       = J(idxJ);
                Cwa     = bsxfun(@times, J, Cwz);
            else
                 J      = J(idxJ);
                Cwa     = J.*Cwz;
            end
        end
        function Caa    = covarianceCaa(J, Cza, idxJ, gpu)
            if gpu
                J       = J(idxJ);
                Caa     = bsxfun(@times, J, Cza);
            else
                J       = J(idxJ);
                Caa     = J.*Cza;
            end
        end
        function Sa     = covarianceSa(J, Sz, gpu)
            if gpu
                Sa      = bsxfun(@times, J, Sz);
                Sa      = bsxfun(@times, Sa, J);
            else
                Sa      = J.*Sz.*J;
            end
        end
    end
end