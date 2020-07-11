clc
clear all
close all
load logmap.mat
addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
addpath('C:\Users\Keerthitheja\Desktop\SVM\svm_toolbox_1\svm');
order = 10;
X = windowize (Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );
gam = 10;
sig2 = 10;
[alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
Xs = Z(end - order +1: end , 1);
nb = 50;
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
figure ;
hold on;
h1 = plot (Ztest , 'k', 'MarkerSize', 20);
h2 = plot ( prediction , 'r', 'MarkerSize', 20);
hold off;
title(['LS-SVM : True VS Predictions with gam = ', num2str(gam), ', sigma = ',...
         num2str(sig2),', order = ',num2str(order)])
legend('Ground Truth','Prediction')
set(h1, 'linewidth', 3)
set(h2, 'linewidth', 3)
set(gca, 'FontSize',20)

gammaList = [0.1 1 5 10 15 20 25 100 1000 10000];
sigmaList = [0.01 0.1 0.5 1 10 15];
orderList = [5:20];
MAE_Values = [];
% for gam = gammaList
%     for sig2 = sigmaList
%         for order = orderList
%             X = windowize (Z, 1:( order + 1));
%             Y = X(:, end);
%             X = X(:, 1: order );
%             Xs = Z(end - order +1: end , 1);
%             prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
%             mae = mean(abs(prediction-Ztest));
%             MAE_Values = [MAE_Values ; gam, sig2, order, mae*100 ];
%             if(mae*100 <= 15)
%                 figure ;
%                 hold on;
%                 h1 = plot (Ztest , 'k', 'MarkerSize', 20);
%                 h2 = plot ( prediction , 'r', 'MarkerSize', 20);
%                 title(['LS-SVM : Ground Truth VS Predictions with gam = ', num2str(gam), ', sigma = ',...
%                 num2str(sig2),', order = ',num2str(order), ', MAE = ', num2str(mae*100),' (%)'])
%                 hold off;
%                 legend('Ground Truth','Prediction')
%                 set(h1, 'linewidth', 3)
%                 set(h2, 'linewidth', 3)
%                 set(gca, 'FontSize',20)
%             end
%         end
%     end
% end

orderList = [1:50];

for order = orderList
    for i = 1:15
        display(order)
        X = windowize (Z, 1:( order + 1));
        Y = X(:, end);
        X = X(:, 1: order );
        Xs = Z(end - order +1: end , 1);
        [gam ,sig2 , cost ] = tunelssvm ({ X , Y , 'f', [], [], ...
        'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'mae'});
        prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
        mae = mean(abs(prediction-Ztest));
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