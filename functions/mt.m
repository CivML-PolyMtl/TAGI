%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         mt
% Description:  Metric for performance evaluation
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 8, 2019
% Updated:      January 23, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef mt
    methods (Static)
        function e = computeError(y, ypred)
            e = mean(sqrt(mean((y-ypred).^2)));
        end
        function e  = errorRate(ytrue, ypred)
            idx_true     = ytrue;
            [~,idx_pred] = max(ypred);
            idx_pred     = idx_pred-1;
            e            = idx_true~=idx_pred;  
        end
        function LL = loglik(y, ypred, Vpred)
            d = size(y, 2);
            if d == 1
                LL = mean(-0.5*log(2*pi*Vpred) - (0.5*(y-ypred).^2)./Vpred);
            else
                LL = mean(-d/2*log(2*pi) - 0.5*log(prod(Vpred, 2)) - sum((0.5*(y-ypred).^2)./Vpred, 2)); 
            end
        end
        function weight     = probFromloglik(loglik)
            maxlogpdf       = max(loglik);
            w_1             = bsxfun(@minus,loglik,maxlogpdf);
            w_2             = log(sum(exp(w_1)));
            w_3             = bsxfun(@minus,w_1,w_2);
            weight          = exp(w_3);
        end
    end
end