clear;
close all; 
clc;


load Tc
load SM
N = size(TC,1); %number of time samples 
nV = sqrt(size(SM,2)); %sqrt of number of voxels
x1 = nV; x2 = nV; 
nSRCS = size(TC,2); %number of sources
nIter = 30; %algorithm iterations
tstd  = sqrt(0.9); %0.6 is the varaince
sstd  = sqrt(0.01);
Dp = dctbases(N,N); %dct basis dictionary



rng('default');
rng('shuffle') % random number generator
Y= (TC+tstd*randn(N,nSRCS))*(SM+sstd*randn(nSRCS,nV^2));
Y= Y-repmat(mean(Y),size(Y,1),1);
K =8;


%% ICA
tic
[G,~,~] = svds(Y,K);
Ss = G'*Y;
[SSs,~,~] = fastica(Ss, 'numOfIC', K,'approach','symm', 'g', 'tanh','verbose', 'off');
X{1} = SSs;
D{1} = Y*SSs';
toc

%% PMD
tic
K1 = 24; K2 = 16;
lambda =0.5; gamma=lambda*ones(1,K1);
tmpX=GPower(Y,gamma,K1,'l1',0);
Y2 = Y*tmpX;
Y2 = Y2*diag(1./sqrt(sum(Y2.*Y2)));
lambda2 = 0.5;
[Wx1,~,~] = pmd_rankK(Y',Y2',K2,lambda2);
X{2} = Wx1';
D{2} = Y*X{2}';
toc

%% SICA-EBM
tic
[G,~,~] = svds(Y,K);
Ss = G'*Y;
[WW,~,~,~,~] = ICA_EBM_Sparse(Ss,50,10^5); %0.7
D{3} = Y*(real(WW)*Ss)'; D{3} = D{3} * diag(1./sqrt(sum(D{3}.*D{3})));
X{3} = WW*Ss;
toc

%% ssBSS
tic
params1.K = K;
params1.P = 8; %8
params1.lam1 = 6; %6
params1.zeta1 = 60;
params1.Kp = 120;
params1.nIter = nIter;
params1.alpha = 10^-8; %1e-9
[D{4},X{4},~,~]=ssBSS_pre(Y,Dp,params1,TC,SM); %_mod
toc

%% SICA-L
tic
spa = 0.01; %0.015
[D{5},X{5},U,Err]=LSICA(Y,K,spa,nIter,TC,SM);  D{5} = D{5} * diag(1./sqrt(sum(D{5}.*D{5}))); 
toc
D{6} = D{5};
X{6} = X{5};

%% statistical_properties_BSS

Dp = dctbases(N,N); %dct basis dictionary
% Dp = create_enhanced_dct_dictionary(300, 120);
tic
tt = size(Y,2)/size(Y,1);
[X{6}, D{6},err] = BDMS(Y, K , 12, 30, tt*0.4,tt*0.05,tt*0.035,tt*0.7, Dp(:,1:120), 60, TC, SM); %0.4,0.05,0.035,0.7 Dp(:,1:150), 60,
% weights = [0.4,0.05,0.035,0.7];
% [X{6}, D{6},err] = SBSS_pareto(Y, K , 12, 30, weights, TC, SM); %0.4,0.05,0.035,0.7

figure; plot(err)
toc


% %%
% [X{6}, D{6}, convergence_errors, pareto_front, selected_solution] = ...
%     multiobjective_fmri_bss(Y, K, 3, 30, TC, SM, 20, 'weighted');


%%
% [W_pareto, Y_pareto, objectives_pareto] = multiobjective_bss(Y, 1);
% D{6} = W_pareto{8};
% X{6} = Y_pareto{15};


%%
nA=7; 
sD{1} = TC;
sX{1} = SM;
for jj =1:nA-1
 [sD{jj+1},sX{jj+1},ind]=sort_TSandSM_spatial(TC,SM,D{jj},X{jj},nSRCS);
%     [sD{jj+1},sX{jj+1},ind]=sort_TSandSM_temporal(TC,D{jj},X{jj});
for ii =1:nSRCS
 TCcorr(jj+1,ii,1) =abs(corr(TC(:,ii),D{jj}(:,ind(ii))));
 SMcorr(jj+1,ii,1) =abs(corr(SM(ii,:)',X{jj}(ind(ii),:)'));
end
end

ccTC = mean(TCcorr(:,:,1),3)
ccSM = mean(SMcorr(:,:,1),3)
TT(1,:) = mean(ccTC');
TT(2,:) = mean(ccSM');

TT

f = figure;
f.Position = [170 120 1500 850];
for j = 1:nSRCS % Sources in rows
    row_base = (j-1) * 14; % Reduced from 15 to 14 to reduce gap
    for i = 1:nA % Methods in columns
        % Time series (swapped to first position)
        subplot_tight(nSRCS+1, 14, row_base + i, [0.02 0.01]);
        plot(zscore(sD{i}(:,j)), 'b', 'LineWidth', 0.25);
        axis tight;
        ylim([-3 3]);
        ax = gca;
        ax.XTick = [1 length(sD{i}(:,j))/2 length(sD{i}(:,j))];
        ax.XTickLabel = {'0', num2str(round(length(sD{i}(:,j))/2)), num2str(length(sD{i}(:,j))-1)};
        set(gca,'YTickLabel','')
        grid on;
        % Spatial maps (swapped to second position)
        subplot_tight(nSRCS+1, 14, row_base + 7 + i, [0.02 0.01]); % Changed from 8 to 7
        imagesc(flipdim(reshape(abs(zscore(sX{i}(j,:))), x1, x2), 1));
        colormap('hot');
        set(gca,'XTickLabel','')
        set(gca,'YTickLabel','')
    end
end
% Bottom row for labels and metrics - reduced gap
bottom_row_base = nSRCS * 14; % Changed from 15 to 14
for i = 1:nA
    % Labels and temporal correlation under time series (now first position)
    subplot_tight(nSRCS+1, 14, bottom_row_base + i, [0.04 0.01]);
    if i == 1
        text(0.5, 0.5, '(a)', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
    else
        letter = sprintf('(%s)', char('a' + i - 1));
        temp_corr = sprintf('γ_T=%.2f', TT(1,i));
        text(0.5, 0.7, letter, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
        text(0.5, 0.3, temp_corr, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
    end
    axis off;
    % Labels and spatial correlation under spatial maps (now second position)
    subplot_tight(nSRCS+1, 14, bottom_row_base + 7 + i, [0.04 0.01]); % Changed from 8 to 7
    if i == 1
        text(0.5, 0.5, '(a)', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
    else
        letter = sprintf('(%s)', char('a' + i - 1));
        spatial_corr = sprintf('γ_S=%.2f', TT(2,i));
        text(0.5, 0.7, letter, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
        text(0.5, 0.3, spatial_corr, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
    end
    axis off;
end




f = figure;
f.Position = [170 120 1200 300]; % Adjusted width for nSRCS columns

% First row: Time series for all sources
for j = 1:nSRCS
    subplot_tight(2, nSRCS, j, [0.05 0.02]);
    plot(zscore(sD{1}(:,j)), 'b', 'LineWidth', 0.25);
    axis tight;
    ylim([-3 3]);
    ax = gca;
    ax.XTick = [1 length(sD{1}(:,j))/2 length(sD{1}(:,j))];
    ax.XTickLabel = {'0', num2str(round(length(sD{1}(:,j))/2)), num2str(length(sD{1}(:,j))-1)};
%     set(gca,'YTickLabel','')
    xlabel('Time'); % Add x-axis label
    grid on;
end

% Second row: Spatial maps for all sources
for j = 1:nSRCS
    subplot_tight(2, nSRCS, nSRCS + j, [0.05 0.02]);
    imagesc(flipdim(reshape(abs(zscore(sX{1}(j,:))), x1, x2), 1));
    colormap('hot');
%     set(gca,'XTickLabel','')
%     set(gca,'YTickLabel','')
end

exportgraphics(gcf,'khali2.png','Resolution',300)