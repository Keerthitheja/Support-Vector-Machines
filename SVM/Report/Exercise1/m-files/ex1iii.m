addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
clc
clear all
close all
load iris.mat;

gammaList = [1, 10, 1000, 10000, 100000, 1000000];
sig2List = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1 , 10, 100, 1000, 10000];
performance = [] ;
for gam = gammaList,
    performance_ = [];
    for sig2 = sig2List,
        perf = rsplitvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, 0.80 , 'misclass');
        performance_ = [performance_ ; perf ] ;
    end
    performance = [performance; performance_] ;
end

X = log(sig2List);
Y = log(gammaList);
Z = reshape(performance,length(X),length(Y)) ;
surf(X,Y,Z)
xlabel('X --> log(sig2List)')
ylabel('Y --> log(gammaList)')
zlabel('Z --> Performance')