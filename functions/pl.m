%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         pl
% Description:  plot figurue for TAGI
% Authors:      James-A. Goulet & Luong-Ha Nguyen
% Created:      November 3, 2019
% Updated:      January 23, 2020
% Contact:      james.goulet@polymtl.ca & luongha.nguyen@gmail.com
% Copyright (c) 2020 James-A. Goulet & Luong-Ha nguyen  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef pl
    methods (Static)
        function plotClassProb(prob_class, y_obs)
            y_test=y_obs;
            %correct_class=prob_class(metric.errorRate==0,1:10);
            %wrong_class=prob_class(metric.errorRate==1,1:10);
            
            pr=sortrows([prob_class(:,1:10),y_test],11);
            idx_y=0;
            for i=0:9
                idx=find(pr(:,11)==i);
                pr(idx,:)=sortrows(pr(idx,:),-(i+1));
                idx_y=[idx_y,idx_y(end)+numel(idx)];
            end
            
            p_class=[0.1:0.01:0.99 0.995 0.997 0.998 0.999];
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
            
            
            figure('Position', [0 0 450 100]);
            subplot(1,3,1:2)
            h=imagesc(pr(:,1:10)',[0,1])
            
            colormap(jet)
            %colorbar
            ax = gca;
            ax.XTick = idx_y;
            ax.YTick = [-0.5:1:10.5];
            
            ax.XColor = [1 1 1];
            ax.YColor = [1 1 1];
            
            
            ax.GridColor = [1 1 1];
            ax.GridAlpha = 1;
            xticklabels({})
            yticklabels({})
            %ylabel('True labels (0-9)')
            %xlabel('Test set labels (0-9)')
            grid on
            
            
            
            subplot(1,3,3)
            h=area(p_class,[stat]')
            h(1).FaceColor = 'green';
            h(2).FaceColor = [1,1,0.1];
            h(3).FaceColor = 'red';
            xlim([0.1,0.999]);
            ax = gca;
            ax.XTick = [0.1 0.5 0.9];
            %ax.YTick = [0 1];
            
            %xticklabels({})
            %xlabel('Threshold Pr')
            ylabel('$\Pr(~)$','Interpreter','Latex')
        end
    end
end