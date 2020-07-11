clc
clear all
close all
X1 = randn(50,2)+1;
X2 = randn(51,2)-1;
Y1 = ones (50 ,1);
Y2 = -ones (51 ,1);
figure(100),
plot(X1(:,1),X1(:,2),'ro')
hold on
plot(X2(:,1),X2(:,2),'bo')
hold on
figure(2)
histogram(X1)
hold on
histogram(X2)
data = [X1;X2];
dataClass = [Y1;Y2];
svmModel = fitcsvm(data,dataClass, 'ClassNames',[1,-1]);
x = linspace(-5,5);
y = linspace(-5,5);
[XX,YY] = meshgrid(x,y);
pred = [XX(:),YY(:)];
p = predict(svmModel,pred);
figure(3)
gscatter(pred(:,1),pred(:,2),p)
f = @(x)-(svmModel.Beta(1)*x+svmModel.Bias)/svmModel.Beta(2);
y = f(x);
plot(x,y,'g--','LineWidth',2,'DisplayName','Boundary')
hold off
figure(4),
hold on
gscatter(data(:,1),data(:,2),dataClass(:,1))
hold on
plot(x,y,'g--','LineWidth',2,'DisplayName','Boundary')
hold off
