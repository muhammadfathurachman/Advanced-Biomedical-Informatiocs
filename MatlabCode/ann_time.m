clear;
load datasetckdna.mat
%% KNN Imputation for NA Values
K = 3;
tempdataset =  datasetckdna;
tempdataset =  knnimpute(tempdataset,K);
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

class = tempdataset(:,25);
index_cross = crossvalind('Kfold', class, 10);

result = zeros(1,6);
hiddenNode = [49:2:101];
[row,col] = size(hiddenNode);

counter = 1;
REPEATED = 100;
for s = 1:10
index_cross = crossvalind('Kfold', class, 10);
    for i = 1 : 10
    test = (index_cross == i);
    training = ~test;
    
    %Training Data
    X_TRAIN = tempdataset(training,2:25)';
    Y_TRAIN = tempdataset(training,25)';
    Y_TRAIN = binaryVector(Y_TRAIN);
    
    %Testing Data
    X_TEST = tempdataset(test,2:25)';
    Y_TEST = tempdataset(test,25)';
    %Y_TEST = binaryVector(Y_TEST);
  
        net_lm = patternnet(61);
        net_lm.trainFcn = 'trainrp';
        st = cputime;
        [net_lm,tr] = train(net_lm,X_TRAIN, Y_TRAIN);
        result(counter,6) = cputime-st;
        prediction = round(net_lm(X_TEST));
        
        prediction = prediction(1,:)';
        result(counter,2)= 61;
        [result(counter,3), result(counter,4), result(counter,5)] = CM(Y_TEST', prediction);
        counter =  counter+1;
    
    end
end 

