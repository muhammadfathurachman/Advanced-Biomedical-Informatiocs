%% Neural Network Backpropagation
clear;
%% Load Data
load datasetckdna.mat

%% KNN Imputation for NA Values
K = 3;
tempdataset =  datasetckdna;
tempdataset =  knnimpute(tempdataset,K);

%% Normalize Data
for i = 2:25
    minval =  min(tempdataset(:,i));
    maxval = max(tempdataset(:,i));
    for j = 1:386
      tempdataset(j,i) = (tempdataset(j,i)- minval)/(maxval - minval);
    end
end
%% Training Data and Testing
HiddenNeuron = 20;
REPEATED =  1;
N = 10;
class = tempdataset(:,25);
for i = 1 : REPEATED
    index =  crossvalind('Kfold',class, N);
    for j = 1 : N
        test = (index == i);
        train = ~test;
        x_test = tempdataset(test,1:24);
        y_test = tempdataset(test,25);
        x_train = tempdataset(train,1:24);
        y_train = tempdataset(train,25);
        nn = patternnet(HiddenNeuron);
        nn = train(nn,x_train,y_train);
        
        %outputs = net(x_test);
        %errors = gsubtract(outputs,y_test);
       % performance = perform(net,y_test,outputs)
    end
end