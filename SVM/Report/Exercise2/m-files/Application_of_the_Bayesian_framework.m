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
gam = 10; sig2 = 0.4;

crit_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1);
crit_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2);
crit_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3);
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure');
Yest1 = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'}, {alpha, b}, Xtest);

[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [], 'RBF_kernel','ds'}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
display(gam)
display(sig2)
crit_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1);
crit_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2);
crit_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3);
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure');
Yest = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'}, {alpha, b}, Xtest);

mse_test1 = mse(Yest1-Ytest);
display(mse_test1)

mse_test = mse(Yest-Ytest);
display(mse_test)