addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
clc
close all
load iris.mat

gammaList = [0.01 1 5 25 100 1000] ;
sig2List = [0.001, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 10000];
performance = [] ;
for gam = gammaList
    performance_ = [];
    for sig2 = sig2List
        %perf = rsplitvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,...
%'RBF_kernel'}, 0.80 , 'misclass');
        perf = crossvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,...
'RBF_kernel'}, 10 , 'misclass'); 
        performance_ = [performance_ ; perf ] ;
    end
    %figure;
    h_ = plot(log(sig2List), ((performance_)*100), '*-', 'LineWidth', 3, 'MarkerSize', 12); 
    ylim([0 max((performance_)*100)+10])
    xlabel(sprintf('log(sig2)-with-gam = %s', num2str(gam))), ylabel('Performance Error % ');
    set(h_ ,'linewidth',2)
    set(gca,'FontSize',20)
    performance = [performance; performance_] ;
    
end
figure
X = log(sig2List);
Y = log(gammaList);
Z = reshape(performance,length(Y),length(X)) ;
h = surf(X,Y,Z) ;
xlabel('X --> log(sig2List)')
ylabel('Y --> log(gammaList)')
zlabel('Z --> Performance Error')
title('-- k-fold Cross-Validation --')
set(h ,'linewidth',2)
set(gca,'FontSize',20)
X_simplex = [];
X_gridSearch = [];
t1 = tic

for i = 1:8
    [gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
    X_simplex = [X_simplex ; gam,sig2,cost];

end
toc(t1)
%for i = 1:12
 %   [gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
 %   X_gridSearch = [X_gridSearch ; gam,sig2,cost];
%end
%format short g
%X_simplex
%X_gridSearch
gam = 6.0573;
sig2 = 0.19643;
[alpha , b] = trainlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'});
[Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, {alpha , b}, Xtest );
roc( Ylatent , Ytest );

gam = 5;
sigma = [0.01, 1, 10];
for sig2=sigma
    bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');
    colorbar
    set(h ,'linewidth',2)
    set(gca,'FontSize',22)
end
sig2 = 1;
gamma = [0.1 3 15];
for gam=gamma
    bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');
    colorbar
    set(h ,'linewidth',2)
    set(gca,'FontSize',22)
end


