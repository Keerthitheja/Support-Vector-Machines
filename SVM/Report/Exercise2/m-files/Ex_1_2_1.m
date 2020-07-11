clc
clear all
close all
addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
addpath('C:\Users\Keerthitheja\Desktop\SVM\svm_toolbox_1\svm');
X = ( -3:0.01:3)' ;
Y = sinc (X) + 0.1.* randn ( length (X), 1);
Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);
type = 'function estimation';
gamma = [0.1 1 10 10^2 10^3 10^4];
sigma = [0.01 0.1 1 10 100 1000];
comb_ = [];
for gam = gamma
    for sig2 = sigma
        %figure
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
        %plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
        YtestEst = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);
        Diff = abs(Ytest-YtestEst).^2;
        MSE = sum(Diff(:))/numel(Ytest);
        display(MSE*100)
        fprintf('gamma = (%f), sigma = (%f) \n',[gam ;sig2])
        fprintf('\n')
        %plot(Xtest,Ytest,'.', 'MarkerSize', 15);
        %hold on;
        %plot(Xtest,YtestEst,'r-+', 'LineWidth', 2);
        %legend('Ytest','YtestEst');
        %title('gam = 10, sig2 = 0.1');
        comb = [gam sig2 MSE*100];
        comb_ = [comb_ ; comb];
        format short g
        display(MSE)
    end
end
X_simplex = [];
for i = 1:5
    tic
    [gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
    toc
    X_simplex = [X_simplex ; gam,sig2,cost];
end

X_gridsearch = [];
for i = 1:5
    tic
    [gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [], 'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'mse'});
    toc
    X_gridsearch = [X_gridsearch ; gam,sig2,cost];
end
display(X_simplex)
display(X_gridsearch)
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/cos-reg-2', '-dpng');