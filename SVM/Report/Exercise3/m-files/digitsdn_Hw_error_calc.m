clc
clear all
load digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));

noisefactor =1.0;

noise = noisefactor*maxx; % sd for Gaussian noise


Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1; 
for i=1:size(Xtest1,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

Xnt2 = Xtest2; 
for i=1:size(Xtest2,1);
  randn('state', N+i);
  Xnt2(i,:) = Xtest2(i,:) + noise*randn(1,dim);
end

%
% select training set
%
Xtr = X(1:1:end,:);



sig2 =dim*mean(var(Xtr));
sigList = [0.1 0.5 1 2 5 10 15 20 25 30 35 50 75 100];
npcs = [2.^(0:7) 190];
idx = 1 ;
errorList = [];
sig2 =dim*mean(var(Xtr)); % rule of thumb
sigmafactor = 0.01:0.15:0.8;
sig2List=sig2*sigmafactor;
for nb = npcs
    for sig2 = sigList
        [lam,U,scr,omeg,recErrors,out] = kpca(Xtr,'RBF_kernel',sig2,Xnt,'eig',nb);
        recErrors
        errorMean = mean(recErrors);
        errorList = [errorList;nb sig2 errorMean];
    end
end

C = {'k','b','r','g','y','m',[.5 .6 .7],[.8 .2 .6], [.4 .6 .9]};
figure
for i = 1:length(npcs)
   h = plot(log(errorList((i-1)*length(sigList)+1:(i)*length(sigList),2)),...
       errorList((i-1)*length(sigList)+1:(i)*length(sigList),3),...
   'color',C{i},'marker','x');
    set(h,'MarkerSize',15,'linewidth',4)
    xlabel('--- log(sig2) ---');
    ylabel('--- Mean of reconstruction Errors ---');
    title('Digit Image Denoising : log(sig2) VS Reconstruction Error on validation dataset');
    set(gca,'FontSize',15)
    hold on
   %pause(1)
end

% for nb = npcs
%     for sig2 = sigList
%         [lam,U,scr,omeg,recErrors,out] = kpca(Xtr,'RBF_kernel',sig2,Xtr,'eig',240);
%         recErrors
%         errorMean = mean(recErrors);
%         errorList = [errorList;nb sig2 errorMean];
%     end
% end
% 
% C = {'k','b','r','y','m',[.5 .6 .7],[.8 .2 .6], [.4 .1 .5],'g'};
% for i = 1:length(npcs)
%    h = plot(log(errorList((i-1)*length(sigList)+1:(i)*length(sigList),2)),...
%        errorList((i-1)*length(sigList)+1:(i)*length(sigList),3),...
%    'color',C{i},'marker','x');
%     set(h,'MarkerSize',15,'linewidth',4)
%     xlabel('--- log(sig2) ---');
%     ylabel('--- Mean of reconstruction Errors ---');
%     title('Digit Image Denoising : log(sig2) VS Reconstruction Error');
%     set(gca,'FontSize',15)
%     hold on
%    %pause(1)
% end
% % figure
% % for i = 1:length(npcs)
% %    h = plot3(npcs(i)*ones(1,length(sigList)),log(errorList((i-1)*length(sigList)+1:(i)*length(sigList),2)),...
% %        errorList((i-1)*length(sigList)+1:(i)*length(sigList),3),...
% %    'color',C{i},'marker','x');
% %     set(h,'MarkerSize',10,'linewidth',2)
% %    hold on
% %    pause(1)
% % end