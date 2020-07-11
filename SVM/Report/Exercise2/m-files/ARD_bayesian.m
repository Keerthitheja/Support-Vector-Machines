clc
clear all
close all
addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
addpath('C:\Users\Keerthitheja\Desktop\SVM\svm_toolbox_1\svm');
X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
gam_1 = 10; sig2_1 = 0.4;
[ selected_1 , ranking_1 ] = bay_lssvmARD ({X, Y, 'f', gam_1 , sig2_1 });
labels = {'X1' 'X2' 'X3' 'Y'};
figure
data = [X(:,1) X(:,2) X(:,3) Y];

[h,ax] = plotmatrix(data,'MarkerSize',15); 
for i = 1:4                                       % label the plots
  xlabel(ax(4,i), labels{i})
  ylabel(ax(i,1), labels{i})
end
figure 
h1 = plot(ranking_1);
title('Ranking of Most relevant data distribution with randomly selected gamma=10 and sigma=0.4 values')
xlabel('Data distribution Index (1,2,3)-->')
ylabel('Ranking -- >')
set(gca, 'FontSize',20)
set(h1, 'linewidth',2)

% [gam ,sig2 , cost ] = tunelssvm ({ X, Y, 'f', [], [], 'RBF_kernel','ds'}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
% [ selected , ranking ] = bay_lssvmARD ({X, Y, 'f', gam_1 , sig2_1 });
% figure 
% h1 = plot(ranking);
% title('Ranking of Most relevant data distribution with input selection using crossvalidation')
% xlabel('Data distribution Index (1,2,3)-->')
% ylabel('Ranking -- >')
% set(gca, 'FontSize',20)
% set(h1, 'linewidth',2)
