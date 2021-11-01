clear;clc;close all;
addpath(genpath('src'));
%% pick dataset type
dataset='samson';
display_if=true;

switch dataset
    case 'samson'
        load samson_cycunet_result.mat  
        col=95;p=3;L=156;N=col*col;
    case 'jasper'
        load jasper_cycunet_result.mat  
        col=100;p=4;L=198;N=col*col;
end
%%
A=reshape(double(A),p,col*col);
abu_est=reshape(double(abu_est),p,col*col);
Y=reshape(double(Y),L,col*col);

M_est=EndmemberEst(Y,abu_est,300); % estimate endmembers by ||X-MA||, when A is given.
%% Evalution metrics
rmse=sqrt(sum(sum((A-abu_est).^2))/(p*col*col))
[SAD,SADerr] = SadEval(M_est,M)
%% Estimate Abundance
if display_if
    figure
    abu_cube=reshape(abu_est',col,col,p);
    for i=1:p
        subplot(2,p,i)
        imagesc(abu_cube(:,:,i)');axis off;
    end
    colormap(jet);
end

%% GT
if display_if
    gt=reshape(A',col,col,p);
    for i=1:p
        subplot(2,p,i+p)
        imagesc(gt(:,:,i)');axis off
    end
    colormap(jet);
end
