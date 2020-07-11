clc
clear all
close all
load breast.mat

Xtrain = trainset;
Ytrain = labels_train;
Xtest = testset;
Ytest = labels_test;

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
    gamma = [gam_lin, gam_poly, gam_RBF] ;
    sigma2 = [sig2_lin, t(1), sig2_RBF];
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
            [Yest, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);
            err = sum(Yest~=Ytest);
            err_lin = err/length(Ytest)*100 ;
            roc( Zt , Ytest );
            set(gca,'FontSize',20)
        elseif (strcmp(ker,'poly_kernel')== 1)
            degree = t(2)
            [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t(1); degree],ker});
            [Yest, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t(1); degree],ker}, {alpha,b}, Xtest);
            err = sum(Yest~=Ytest);
            err_poly = err/length(Ytest)*100 ;
            roc( Zt , Ytest );
            set(gca,'FontSize',20)
        else
            [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,ker});
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