
%% Time SVM

% KNN Imputation Handle Missing Data
clear;
load datasetckdna.mat;
K = 3;
tempdataset =  datasetckdna;
tempdataset = knnimpute(tempdataset, K);
%Modify Class to Binary
location = find(tempdataset(:,25)== 2);
tempdataset(location,25) = 0;

%% Normalize Data
for i = 2:25
    minval =  min(tempdataset(:,i));
    maxval = max(tempdataset(:,i));
    for j = 1:386
      tempdataset(j,i) = (tempdataset(j,i)- minval)/(maxval - minval);
    end
end

%% Distinct Variable and Class target

%% K-FoldCrossvalidation
REPEATED = 100;
N = 10;
class = tempdataset(:,25);
INDEX = crossvalind('Kfold',class,N);
result_poly = zeros(N,4);
counter = 1;
for j = 1 : REPEATED
    INDEX = crossvalind('Kfold',class,N);
    for i = 1: N
        test = (INDEX == i);
        train = ~test;
        X_test =  tempdataset(test,1:24);
        Y_test =  tempdataset(test,25);
        data_train = tempdataset(train,:);
        X_train = tempdataset(train,1:24);
        Y_train = tempdataset(train,25);
        
        st = cputime;
        svmmodel_poly = fitcsvm(X_train,Y_train,'KernelFunction','polynomial','Standardize',true);
        y_predicted_poly = predict(svmmodel_poly,X_test);
        [result_poly(counter,1),result_poly(counter,2),result_poly(counter,3)] = CM(Y_test, y_predicted_poly);
        result_poly(counter,4) = cputime-st;
        counter = counter+1;
    end
end

result_poly;


