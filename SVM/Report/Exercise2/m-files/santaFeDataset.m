clc
clear all
close all
load santafe.mat
addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
addpath('C:\Users\Keerthitheja\Desktop\SVM\svm_toolbox_1\svm');
order = 50;
gam = 10;
sig2 = 10;
X = windowize (Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );
Xs = Z(end - order +1: end , 1);
nb = 50;
horizon = length(Ztest)-order;
model = trainlssvm({X,Y,'f',gam,sig2, 'RBF_kernel'});
prediction = predict (model, Xs , horizon);
figure ;
hold on;
h1 = plot (Ztest , 'b', 'MarkerSize', 20);
h2 = plot ( prediction , 'r', 'MarkerSize', 20);
hold off;
title(['Santa Fe Dataset: True VS Predictions with gam = ', num2str(gam), ', sigma = ',...
         num2str(sig2),', order = ',num2str(order)])
legend('Ground Truth','Prediction')
set(h1, 'linewidth', 3)
set(h2, 'linewidth', 3)
set(gca, 'FontSize',20)
orderList = [1:30];
MAE_Values = [];
 for order = orderList
     for i = 1:3 
        display(order)
        horizon = length(Ztest)-order;
        X = windowize (Z, 1:( order + 1));
        Y = X(:, end);
        X = X(:, 1: order );
        Xs = Z(end - order +1: end , 1);
        [gam ,sig2 , cost ] = tunelssvm ({ X , Y , 'f', [], [], ...
        'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'mae'});
        prediction = predict ({X, Y, 'f', gam , sig2 }, Ztest(1:order) , horizon);
        mae = mean(abs(prediction-Ztest(order+1:end)));
        MAE_Values = [MAE_Values ; gam, sig2, order, mae*100 ];
        if(mae*100 <= 15)
            figure ;
            hold on;
            h1 = plot (Ztest , 'k', 'MarkerSize', 20);
            h2 = plot ( prediction , 'r', 'MarkerSize', 20);
            title(['LS-SVM : Ground Truth VS Predictions with gam = ', num2str(gam), ', sigma = ',...
            num2str(sig2),', order = ',num2str(order), ', MAE = ', num2str(mae*100),' (%)'])
            hold off;
            legend('Ground Truth','Prediction')
            set(h1, 'linewidth', 3)
            set(h2, 'linewidth', 3)
            set(gca, 'FontSize',20)
        end
    end
end