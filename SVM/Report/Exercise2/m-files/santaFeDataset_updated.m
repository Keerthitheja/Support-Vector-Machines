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
lag_all = 15:1:50;
mae_all = ones(1, length(lag_all));
mape_all = ones(1, length(lag_all));

for i = 1:length(lag_all),
    
    lag = lag_all(i); % lag of the series
    X = windowize(Z,1:(lag+1));
    Y = X(:,end);
    X = X(:,1:lag);
    horizon = length(Ztest)-lag;
    
    [gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'}, ...
    'simplex','crossvalidatelssvm', {10,'mae'});

    model = trainlssvm({X,Y,'f',gam,sig2, 'RBF_kernel'});
    prediction = predict(model,Ztest(1:lag),horizon);
    
    mape_all(i) = mean(abs(prediction-Ztest(lag+1:end))./abs(Ztest(lag+1:end)));
    mae_all(i) = mean(abs(prediction-Ztest(lag+1:end)));
    
    figure;
    h = plot([Ztest(lag+1:end) prediction]);
    xlabel('Time');
    legend('Ground truth Data','Prediction');
    title(['Santa Fe Dataset: True VS Predictions with gam = ', num2str(gam), ', sigma = ',...
         num2str(sig2),', order = ',num2str(lag),', MAE = ', num2str(mape_all(i)*100)]);
    set(h, 'linewidth', 3)
    set(gca, 'FontSize',20)
end
[lag_idx] = find(mape_all == min(mape_all(:)));
mape_all(lag_idx)
figure
h = plot(lag_all(15:end), mape_all(15:end), 'k+-', 'MarkerSize', 15);
xlabel('Order/Lag');
ylabel('Mean Absolute Percentage Error');
title('Santa Fe Dataset: Mean Absolute Percentage Error');
set(gca,'FontSize',20);
set(h,'linewidth',3)
