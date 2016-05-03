%% Time Execute;

clear;
load datasetckdna.mat;

% KNN Imputation Handle Missing Data
% Menentukan Nilai K ?
K = 3;
tempdataset =  datasetckdna;
tempdataset = knnimpute(tempdataset, K);
tempdataset_svm = tempdataset;
%% Move target class, to first column
dataset = tempdataset(:,25);
temp_class = tempdataset(:,25);
temp =  tempdataset(:,1);


tempdataset(:,25) =  temp;
tempdataset(:,1) =  temp_class;

%%Normalize data into 0-1
for i = 2:25
    minval =  min(tempdataset(:,i));
    maxval = max(tempdataset(:,i));
    for j = 1:386
      tempdataset(j,i) = (tempdataset(j,i)- minval)/(maxval - minval);
    end
end

%Perform 10 Fold Crossvalidation
N = 10;
L_HN = 150;
REPEATED  = 100;
hasil_type = 8;
result_radbas = zeros(L_HN,hasil_type);
class = tempdataset(:,1);
%index_cros = crossvalind('Kfold',class,N);
counter = 1;
for c = 1: REPEATED
    index_cross = crossvalind('Kfold', class, N);
        for i = 1:N
            test = (index_cross == i);
            train = ~test;
            [result_radbas(counter,7),result_radbas(counter,8),result_radbas(counter,3),result_radbas(counter,4),result_radbas(counter,5),result_radbas(counter,6)] = elm_r(tempdataset(train,:),tempdataset(test,:),1,89,'radbas');
            counter = counter+1;
        end
end
result_radbas

