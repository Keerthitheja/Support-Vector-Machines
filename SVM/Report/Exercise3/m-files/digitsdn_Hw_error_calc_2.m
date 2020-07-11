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
sigList = [0.01 0.1 1 10 25 50 75 100 1000];
npcs = [2.^(0:7) 190];
idx = 1 ;
errorListFinal = [];
sig2 =dim*mean(var(Xtr)); % rule of thumb
sigmafactor = 0.01:0.15:0.8;
sig2List=sig2*sigmafactor;
recList = [];

for sig2 = sigList
    errorList = [];
    [lam,U,scr,omeg,recErrors,out] = kpca(Xtr,'RBF_kernel',sig2,Xnt,'eig',240,'r');
    recList = [recList recErrors];
    % choose the digits for test
    digs=[0:9]; ndig=length(digs);
    m=2; % Choose the mth data for each digit 

    Xdt=zeros(ndig,dim);

    % which number of eigenvalues of kpca
    npcs = [2.^(0:7) 190];
    lpcs = length(npcs);


    for k=1:lpcs;
        nb_pcs=npcs(k); 
        disp(['nb_pcs = ', num2str(nb_pcs)]); 
        Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
        err= [];
        for i=1:ndig
            dig=digs(i);
            fprintf('digit %d : ', dig)
            xt=Xnt(i,:);   
            Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
            err = [err; sqrt((mean(Xtest1(i,:)-Xdt(i,:))).^2)];
            end % for i
        errorList =  [errorList err];
%     display(sig2);
    end % for k
    errorListFinal = [errorListFinal errorList];
end

C = {'k','b','r','g','y','m',[.5 .6 .7],[.8 .2 .6], [.4 .6 .9]};
figure
for i = 1:length(sigList)
    for j = 1:l0
        errors = errorListFinal(:,(9*i-9)+1:i*9);
        plot3(repmat(log(sigList)', 1, size(errors,2)), repmat(1:9, size(errors,1),1), errors(j,:),'x-');
        hold on
    end
end
            
% for i = 1:length(npcs)
%    h = plot(log(errorList((i-1)*length(sigList)+1:(i)*length(sigList),2)),...
%        errorList((i-1)*length(sigList)+1:(i)*length(sigList),3),...
%    'color',C{i},'marker','x');
%     set(h,'MarkerSize',15,'linewidth',4)
%     xlabel('--- log(sig2) ---');
%     ylabel('--- Mean of reconstruction Errors ---');
%     title('Digit Image Denoising : log(sig2) VS Reconstruction Error on validation dataset');
%     set(gca,'FontSize',15)
%     hold on
%    %pause(1)
% end