clc
clear all
close all
load ripley.mat

h = gscatter(Xtrain(:,1), Xtrain(:,2), Ytrain, 'km', 'xp');
set(h ,'linewidth',2);
set(h, 'MarkerSize', 15);
set(gca,'FontSize',22);
xlabel('X1');
ylabel('X2');
title('Ripley dataset - Training data')
lgd = legend({'Data = X1 , Class = -1','Data = X2, Class = 1'},'Location','Best');
lgd.FontSize = 20;

figure,
h1 = gscatter(Xtest(:,1), Xtest(:,2), Ytest, 'km', 'xp');
set(h1 ,'linewidth',2);
set(h1, 'MarkerSize', 15);
set(gca,'FontSize',22);
xlabel('X1');
ylabel('X2');
title('Ripley dataset - Test data')
lgd1 = legend({'Data = X1 , Class = -1','Data = X2, Class = 1'},'Location','Best');
lgd1.FontSize = 20;

errLinear = [];
errPoly = [];
errRBF = [];
gammaOptimal_lin = [];
sigmaOptimal_lin = [];
gammaOptimal_poly = [];
sigmaOptimal_poly = [];
gammaOptimal_RBF = [];
sigmaOptimal_RBF = [];
for i=1:3
    [gam_lin ,sig2_lin , cost_lin ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [], 'lin_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
    [gam_poly ,t , cost_poly ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [], 'poly_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
    [gam_RBF ,sig2_RBF , cost_RBF ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
    gammaOptimal_lin = [gammaOptimal_lin ;gam_lin ];
    sigmaOptimal_lin = [sigmaOptimal_lin ; sig2_lin];
    gammaOptimal_poly = [gammaOptimal_poly; gam_poly];
    sigmaOptimal_poly = [sigmaOptimal_poly; t(1)];
    gammaOptimal_RBF = [gammaOptimal_RBF; gam_RBF];
    sigmaOptimal_RBF = [sigmaOptimal_RBF; sig2_RBF];
    format short g
    sig2_poly = t(1);
    gamma = [gam_lin, gam_poly, gam_RBF] ;
    sigma2 = [sig2_lin, sig2_poly , sig2_RBF];
    kernel = ["lin_kernel" , "poly_kernel", "RBF_kernel"] ;
    type = 'c' ;
    for i=1:3
        gam = gamma(i)
        if(i > 1)
            sig2 = sigma2(i-1) 
        end
        ker = kernel(i) 
        if(strcmp(ker,'lin_kernel') == 1)
            [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});
            figure; plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
            [Yest, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);
            set(gca,'FontSize',20)
            err = sum(Yest~=Ytest);
            err_lin = err/length(Ytest)*100 ;
            roc( Zt , Ytest );
            set(gca,'FontSize',20)
        elseif (strcmp(ker,'poly_kernel')== 1)
            [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,t,ker});
            figure; plotlssvm({Xtrain,Ytrain,type,gam,[t(1); t(2)],'poly_kernel','preprocess'},{alpha,b});
            set(gca,'FontSize',20)
            [Yest, Zt] = simlssvm({Xtrain,Ytrain,type,gam,t,ker}, {alpha,b}, Xtest);
            err = sum(Yest~=Ytest);
            err_poly = err/length(Ytest)*100 ;
            roc( Zt , Ytest );
            set(gca,'FontSize',20)
        else
            [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,ker});
            figure; plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
            set(gca,'FontSize',20)
            [Yest, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,ker}, {alpha,b}, Xtest);
            err = sum(Yest~=Ytest);
            err_RBF = err/length(Ytest)*100 ; 
            roc( Zt , Ytest );
            set(gca,'FontSize',20)
        end
    end
    errLinear = [errLinear;err_lin ];
    errPoly = [errPoly; err_poly];
    errRBF = [errRBF;err_RBF];
end
 gammaOptimal_lin
 sigmaOptimal_lin
 gammaOptimal_poly
 sigmaOptimal_poly
 gammaOptimal_RBF
 sigmaOptimal_RBF
 errLinear
 errPoly
 errRBF
 close all