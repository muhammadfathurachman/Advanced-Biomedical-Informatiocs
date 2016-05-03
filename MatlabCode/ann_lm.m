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

result = zeros(1,5);
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
    
    for j = 1 : col
        net_lm = patternnet(hiddenNode(1,j));
        net_lm.trainFcn = 'trainscg';
        [net_lm,tr] = train(net_lm,X_TRAIN, Y_TRAIN);
        prediction = round(net_lm(X_TEST));
        
        prediction = prediction(1,:)';
        result(counter,2)= hiddenNode(1,j);
        [result(counter,3), result(counter,4), result(counter,5)] = CM(Y_TEST', prediction);
        counter =  counter+1;
    end
    end
end 

%% Calculate Average : 
[g d] = size(hiddenNode)
mean_resutl = zeros(1,8);
for i = 1 : d
    mean_result(i,1) = hiddenNode(1,i);
    index_find = find(result(:,2) == hiddenNode(1,i));
   
    mean_result(i,1) = hiddenNode(1,i);
    %% Accuracy
    mean_result(i,2) = mean(result(index_find,3));
    %% Sensitivity
    mean_result(i,3) = mean(result(index_find,4));
    %% Specificity
    mean_result(i,4) = mean(result(index_find,5));
     
    %% Standard Deviasi Acc
    mean_result(i,5) = std(result(index_find,3));
    %% Standar Deviasi Sensitivity
    mean_result(i,6) = std(result(index_find,4));
    %% Standar Deviasi Specificity
    mean_result(i,7) = std(result(index_find,5));
    
end

figure
subplot(2,2,1)
plot(mean_result(:,1),mean_result(:,2)) % line plot
xlabel('Hidden Neuron');
ylabel('Accuracy');
title('Accuracy ANN-Resilient Backpropagation')

subplot(2,2,2)
plot(mean_result(:,1),mean_result(:,3)) % line plot
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('Sensitivity ANN-Resilient Backpropagation')

subplot(2,2,3)
plot(mean_result(:,1),mean_result(:,4)) % line plot
xlabel('Hidden Neuron');
ylabel('Specificity');
title('Specificity ANN-Resilient Backpropagation')

subplot(2,2,4)
plot(mean_result(:,1),mean_result(:,2)) % line plot
hold on
plot(mean_result(:,1),mean_result(:,3)) % line plot
hold on
plot(mean_result(:,1),mean_result(:,4)) % line plot
title('Peforma Resilient Backpropagation')
l = legend('Accuracy','Sensitivity','Specificity');