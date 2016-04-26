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
REPEATED = 10;
N = 10;
class = tempdataset(:,25);
INDEX = crossvalind('Kfold',class,N);
result_poly = zeros(N,3);
result_rbf = zeros(N,3);
result_gauss = zeros(N,3);
result_linear = zeros(N,3);
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

        svmmodel_poly = fitcsvm(X_train,Y_train,'KernelFunction','polynomial','Standardize',true);
        svmmodel_rbf = fitcsvm(X_train,Y_train,'KernelFunction','rbf','Standardize',true);
        svmmodel_gauss = fitcsvm(X_train,Y_train,'KernelFunction','gaussian','Standardize',true);
        svmmodel_linear = fitcsvm(X_train,Y_train,'KernelFunction','linear','Standardize',true);
        
        y_predicted_poly = predict(svmmodel_poly,X_test);
        y_predicted_rbf = predict(svmmodel_rbf,X_test);
        y_predicted_gauss = predict(svmmodel_gauss,X_test);
        y_predicted_linear = predict(svmmodel_linear,X_test);
        
        [result_poly(counter,1),result_poly(counter,2),result_poly(counter,3)] = CM(Y_test, y_predicted_poly);
        [result_rbf(counter,1),result_rbf(counter,2),result_rbf(counter,3)] = CM(Y_test, y_predicted_rbf);
        [result_gauss(counter,1),result_gauss(counter,2),result_gauss(counter,3)] = CM(Y_test, y_predicted_gauss);
        [result_linear(counter,1),result_linear(counter,2),result_linear(counter,3)] = CM(Y_test, y_predicted_linear);
        
        counter = counter+1;
    end
end

result_poly;
result_rbf;
result_gauss;
result_linear;

%% Bar Hasil Accuracy
figure;
hasil_accuracy = [mean(result_poly(:,1)),mean(result_rbf(:,1)),mean(result_gauss(:,1)),mean(result_linear(:,1))];
subplot(2,2,1);
bar(hasil_accuracy);
title('Accuracy');
Labels = {'Polynomial', 'Radial Basis', 'Gaussian', 'Linear'};
set(gca, 'XTick', 1:4, 'XTickLabel', Labels);

subplot(2,2,2);
hasil_sensitivity = [mean(result_poly(:,2)),mean(result_rbf(:,2)),mean(result_gauss(:,2)),mean(result_linear(:,2))];
bar(hasil_sensitivity);
title('Sensitivity');
Labels = {'Polynomial', 'Radial Basis', 'Gaussian', 'Linear'};
set(gca, 'XTick', 1:4, 'XTickLabel', Labels);

subplot(2,2,3);
hasil_specificity = [mean(result_poly(:,3)),mean(result_rbf(:,3)),mean(result_gauss(:,3)),mean(result_linear(:,3))];
bar(hasil_specificity);
title('Specificity')
Labels = {'Polynomial', 'Radial Basis', 'Gaussian', 'Linear'};
set(gca, 'XTick', 1:4, 'XTickLabel', Labels);
