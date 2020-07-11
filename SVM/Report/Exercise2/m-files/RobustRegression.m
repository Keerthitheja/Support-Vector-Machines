clc
clear all
close all
addpath('C:\Users\Keerthitheja\Desktop\SVM\LSSVMlab');
addpath('C:\Users\Keerthitheja\Desktop\SVM\svm_toolbox_1\svm');
X = ( -6:0.2:6)';
Y = sinc (X) + 0.1.* rand ( size (X));

model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
model = tunelssvm (model , 'simplex', 'crossvalidatelssvm' , {10 , 'mae'});
figure
plotlssvm ( model );

out = [15 17 19];
Y( out) = 0.7+0.3* rand ( size ( out));
out = [41 44 46];
Y( out) = 1.5+0.2* rand ( size ( out));

model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
model = tunelssvm (model , 'simplex', 'crossvalidatelssvm' , {10 , 'mae'});
figure
plotlssvm ( model );

funcList = ["whuber", "wlogistic","wmyriad", "whampel"];
type = 'f';
duration = []
for func=funcList
    model = initlssvm(X,Y,type,[],[],'RBF_kernel');
    costFun = 'rcrossvalidatelssvm';
    model = tunelssvm(model,'simplex',costFun,{10,'mae'},func);
    tuneModel = model;
    model = robustlssvm(model);
    figure
    plotlssvm(model);
    duration = [duration;model.duration];
end