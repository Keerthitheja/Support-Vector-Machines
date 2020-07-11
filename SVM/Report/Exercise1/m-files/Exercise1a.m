clc
clear all
close all
X1 = randn(50,2)+1;
X2 = randn(51,2)-1;
Y1 = ones (50 ,1);
Y2 = -ones (51 ,1);
figure(1),
h1 = plot(X1(:,1),X1(:,2),'ro','MarkerSize',15);
hold on
set(h1 ,'linewidth',2)
h2 = plot(X2(:,1),X2(:,2),'bo','MarkerSize',15);
hold on
set(h2 ,'linewidth',2)
set(gca,'FontSize',20)
data = [X1;X2];
dataClass = [Y1;Y2];
svmModel = fitcsvm(data,dataClass, 'ClassNames',[1,-1]);
a_min = min(X1);
b_min = min(X2);
a_max = max(X1);
b_max = max(X2);
x = linspace((min(min(a_min,b_min))) , (max(max(a_max,b_max)))) ;
figure(1)
f = @(x)-(svmModel.Beta(1)*x+svmModel.Bias)/svmModel.Beta(2);
y = f(x);
hold on
h = plot(x,y,'g--','LineWidth',4);
hold on
%h_ = plot(svmModel.SupportVectors(:,1),svmModel.SupportVectors(:,2),'k*');
h_ = plot(data(svmModel.IsSupportVector,1),...
    data(svmModel.IsSupportVector,2),'ko','MarkerSize',20);
set(h_ ,'linewidth',4)
lgd = legend({'Data = X1 , Class = 1','Data = X2, Class = -1',...
    'Decision Boundary','Support Vectors'},'Location','Best');
lgd.FontSize = 20;
xlabel('X1');
ylabel('X2');
hold off
