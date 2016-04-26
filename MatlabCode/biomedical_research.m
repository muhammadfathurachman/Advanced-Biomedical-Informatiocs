clear;
load datasetckdna.mat;

% KNN Imputation Handle Missing Data
% Menentukan Nilai K ?
K = 3;
tempdataset =  datasetckdna;
tempdataset = knnimpute(tempdataset, K);

%% Move target class, to first column
dataset = tempdataset(:,25);
temp_class = tempdataset(:,25);
temp =  tempdataset(:,1);
tempdataset(:,25) =  temp;
tempdataset(:,1) =  temp_class;

%Normalize data into 0-1
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
REPEATED  = 10;
hasil_type = 8;
result_sin = zeros(L_HN*N,hasil_type);
result_sig = zeros(L_HN*N,hasil_type);
result_hardlim = zeros(L_HN,hasil_type);
result_tribas = zeros(L_HN,hasil_type);
result_radbas = zeros(L_HN,hasil_type);
class = tempdataset(:,1);
%index_cros = crossvalind('Kfold',class,N);
n = 1;
counter = 1;

for c = 1: REPEATED
    index_cross = crossvalind('Kfold', class, N);
        for i = 1:N
            test = (index_cross == i);
            train = ~test;
            n = 49;
            for k = 50 : (L_HN/2)
             n = n+2;
             
             %Save fold and number of hidden neuron that belong to sin result
             result_sin(counter,1) = n;
             result_sin(counter,2) = i;
             %Save fold and number of hidden neuron that belong to sig result
             result_sig(counter,1) = n;
             result_sig(counter,2) = i;

             %Save fold and number of hidden neuron that belong to Hardlim Result;
             result_hardlim(counter,1) = n;
             result_hardlim(counter,2) = i;

             %Save fold and number of hidden neuron that belong to Radial Basis
             result_radbas(counter,1) = n;
             result_radbas(counter,2) = i;

             %Save fold and number of hidden neuron that belong to Tribas
             result_tribas(counter,1) = n;
             result_tribas(counter,2) = i;

             % ELM Output :
             % 1.Training Time, 2.Testing Time, 3.Sensitivity, 
             % 4.Specificity, 5. Training Accuracy, 6. Testing Accuracy
             
             [result_sin(counter,7),result_sin(counter,8),result_sin(counter,3),result_sin(counter,4),result_sin(counter,5),result_sin(counter,6)] = ELM(tempdataset(train,:),tempdataset(test,:),1,n,'sin');
             [result_sig(counter,7),result_sig(counter,8),result_sig(counter,3),result_sig(counter,4),result_sig(counter,5),result_sig(counter,6)] = ELM(tempdataset(train,:),tempdataset(test,:),1,n,'sig');
             [result_hardlim(counter,7),result_hardlim(counter,8),result_hardlim(counter,3),result_hardlim(counter,4),result_hardlim(counter,5),result_hardlim(counter,6)] = ELM(tempdataset(train,:),tempdataset(test,:),1,n,'hardlim');
             [result_radbas(counter,7),result_radbas(counter,8),result_radbas(counter,3),result_radbas(counter,4),result_radbas(counter,5),result_radbas(counter,6)] = ELM(tempdataset(train,:),tempdataset(test,:),1,n,'radbas');
             [result_tribas(counter,7),result_tribas(counter,8),result_tribas(counter,3),result_tribas(counter,4),result_tribas(counter,5),result_tribas(counter,6)] = ELM(tempdataset(train,:),tempdataset(test,:),1,n,'tribas');
             counter = counter+1;
            end
      end
end

result_tribas(isnan(result_tribas)) = 0;

%% Count Accuracy for each Activation Function

mean_sinresult = zeros(1,8);
mean_sigresult = zeros(1,8);
mean_hardlimresult = zeros(1,8);
mean_radbasresult = zeros(1,8);
mean_tribasresult = zeros(1,8);

a = 49;
for z =  1 : 26
    a =  a+2;
    index_sin = find(result_sin(:,1)==a);
    index_sig = find(result_sig(:,1)==a);
    index_hard = find(result_hardlim(:,1)==a);
    index_radbas = find(result_radbas(:,1)==a);
    index_tribas = find(result_tribas(:,1)==a);
    
    mean_sinresult(z,1) = a;
    mean_sigresult(z,1) = a;
    mean_hardlimresult(z,1) = a;
    mean_radbasresult(z,1) = a;
    mean_tribasresult(z,1) = a;
    
    % Sin Accuracy
    mean_sinresult(z,2) = mean(result_sin(index_sin,5));
    mean_sinresult(z,3) = mean(result_sin(index_sin,6));
    % Sin Sensitivity
    mean_sinresult(z,4) = mean(result_sin(index_sin,3));
    % Sin Specificity
    mean_sinresult(z,5) = mean(result_sin(index_sin,4));
    % Standard Deviasi Accuracy
    mean_sinresult(z,6) = std(result_sin(index_sin,6));
    % Standard Deviasi Sensitivity
    mean_sinresult(z,7) = std(result_sin(index_sin,3));
    % Standard Deviasi Specificity
    mean_sinresult(z,8) = std(result_sin(index_sin,4));
    
    % Sig Accuracy
    mean_sigresult(z,2) = mean(result_sig(index_sig,5));
    mean_sigresult(z,3) = mean(result_sig(index_sig,6));
    % Sig Sensitivity
    mean_sigresult(z,4) = mean(result_sig(index_sig,3));
    % Sig Specificity
    mean_sigresult(z,5) = mean(result_sig(index_sig,4));
    % Standard Deviasi Accuracy
    mean_sigresult(z,6) = std(result_sig(index_sig,6));
    % Standard Deviasi Sensitivity
    mean_sigresult(z,7) = std(result_sig(index_sig,3));
    % Standard Deviasi Specificity
    mean_sigresult(z,8) = std(result_sig(index_sig,4));
    
   
    % Hardlim Accuracy
    mean_hardlimresult(z,2) = mean(result_hardlim(index_hard,5));
    mean_hardlimresult(z,3) = mean(result_hardlim(index_hard,6));
    % Hardlim Sensitivity
    mean_hardlimresult(z,4) = mean(result_hardlim(index_hard,3));
    % Hardlim Specificity
    mean_hardlimresult(z,5) = mean(result_hardlim(index_hard,4));
    % Standard Deviasi Accuracy Testing
    mean_hardlimresult(z,6) = std(result_hardlim(index_hard,6));
    % Standard Deviasi Sensitivity Testing
    mean_hardlimresult(z,7) = std(result_hardlim(index_hard,3));
    % Standard Deviasi Specificity Testing
    mean_hardlimresult(z,8) = std(result_hardlim(index_hard,4));
    
    
    % Radial Basis Accuracy
    mean_radbasresult(z,2) = mean(result_radbas(index_radbas,5));
    mean_radbasresult(z,3) = mean(result_radbas(index_radbas,6));
    % Radial Basis Sensitivity
    mean_radbasresult(z,4) = mean(result_radbas(index_radbas,3));
    %Radial Basis Specificity
    mean_radbasresult(z,5) = mean(result_radbas(index_radbas,4));
    % Standard Deviasi Accuracy Testing
    mean_radbasresult(z,6) = std(result_radbas(index_radbas,6));
    % Standard Deviasi Sensitivity Testing
    mean_radbasresult(z,7) = std(result_radbas(index_radbas,3));
    % Standard Deviasi Specificity Testing
    mean_radbasresult(z,8) = std(result_radbas(index_radbas,4));
     
    % Tribas Accuracy
    mean_tribasresult(z,2) =  mean(result_tribas(index_tribas,5));
    mean_tribasresult(z,3) = mean(result_tribas(index_tribas,6));
    % Tribas Sensitivity 
    mean_tribasresult(z,4) = mean(result_tribas(index_tribas,3));
    % Tribas Specificity 
    mean_tribasresult(z,5) = mean(result_tribas(index_tribas,4));
    % Standard Deviasi Accuracy Testing
    mean_tribasresult(z,6) = std(result_tribas(index_tribas,6));
    % Standard Deviasi Sensitivity Testing
    mean_tribasresult(z,7) = std(result_tribas(index_tribas,3));
    % Standard Deviasi Specificity Testing
    mean_tribasresult(z,8) = std(result_tribas(index_tribas,4));
end

%% Plot Hasil Akurasi
figure
subplot(3,2,1)       % add first plot in 2 x 2 grid
plot(mean_sinresult(:,1),mean_sinresult(:,3)) % line plot
xlabel('Hidden Neuron');
ylabel('Accuracy');
title('ELM dengan Fungsi Aktivasi Sin')

subplot(3,2,2)       % add second plot in 2 x 2 grid
plot(mean_sigresult(:,1),mean_sigresult(:,3))        % scatter plot
xlabel('Hidden Neuron');
ylabel('Accuracy');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Sigmoid')

subplot(3,2,3)       % add third plot in 2 x 2 grid
plot(mean_hardlimresult(:,1),mean_hardlimresult(:,3)) % line plot % stem plot
xlabel('Hidden Neuron');
ylabel('Accuracy');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Hardlim')

subplot(3,2,4)       % add fourth plot in 2 x 2 grid
plot(mean_radbasresult(:,1),mean_radbasresult(:,3))
xlabel('Hidden Neuron');
ylabel('Accuracy');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Radial Basis')

subplot(3,2,5)       % add fourth plot in 2 x 2 grid
plot(mean_tribasresult(:,1),mean_tribasresult(:,3))
xlabel('Hidden Neuron');
ylabel('Accuracy');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Tribas')

subplot(3,2,6)
plot(mean_sinresult(:,1),mean_sinresult(:,3))
hold on
plot(mean_sigresult(:,1),mean_sigresult(:,3))   
hold on
plot(mean_hardlimresult(:,1),mean_hardlimresult(:,3)) % line plot % stem plot
hold on
plot(mean_radbasresult(:,1),mean_radbasresult(:,3))
hold on
plot(mean_tribasresult(:,1),mean_tribasresult(:,3))
xlabel('Hidden Neuron');
ylabel('Accuracy');
l = legend('sin','sig','Hardlim','Radbas','Tribas');


%% Plot Hasil Sensitivity
figure
subplot(3,2,1)       % add first plot in 2 x 2 grid
plot(mean_sinresult(:,1),mean_sinresult(:,4)) % line plot
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('ELM dengan Fungsi Aktivasi Sin')

subplot(3,2,2)       % add second plot in 2 x 2 grid
plot(mean_sigresult(:,1),mean_sigresult(:,4))        % scatter plot
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('ELM dengan Fungsi Aktivasi Sigmoid')

subplot(3,2,3)       % add third plot in 2 x 2 grid
plot(mean_hardlimresult(:,1),mean_hardlimresult(:,4)) % line plot % stem plot
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('ELM dengan Fungsi Aktivasi Hardlim')

subplot(3,2,4)       % add fourth plot in 2 x 2 grid
plot(mean_radbasresult(:,1),mean_radbasresult(:,4))
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('ELM dengan Fungsi Aktivasi Radial Basis')

subplot(3,2,5)       % add fourth plot in 2 x 2 grid
plot(mean_tribasresult(:,1),mean_tribasresult(:,4))
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('ELM dengan Fungsi Aktivasi Tribas');

subplot(3,2,6)
plot(mean_sinresult(:,1),mean_sinresult(:,4))
hold on
plot(mean_sigresult(:,1),mean_sigresult(:,4))   
hold on
plot(mean_hardlimresult(:,1),mean_hardlimresult(:,4)) % line plot % stem plot
hold on
plot(mean_radbasresult(:,1),mean_radbasresult(:,4))
hold on
plot(mean_tribasresult(:,1),mean_tribasresult(:,4))
xlabel('Hidden Neuron');
ylabel('Sensitivity');
title('Average')
l = legend('sin','sig','Hardlim','Radbas','Tribas');


%% Plot Hasil Specificity

figure
subplot(3,2,1)       % add first plot in 2 x 2 grid
plot(mean_sinresult(:,1),mean_sinresult(:,5)) % line plot
xlabel('Hidden Neuron');
ylabel('Specificity');
title('ELM dengan Fungsi Aktivasi Sin')

subplot(3,2,2)       % add second plot in 2 x 2 grid
plot(mean_sigresult(:,1),mean_sigresult(:,5))        % scatter plot
xlabel('Hidden Neuron');
ylabel('Specificity');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Sigmoid')

subplot(3,2,3)       % add third plot in 2 x 2 grid
plot(mean_hardlimresult(:,1),mean_hardlimresult(:,5)) % line plot % stem plot
xlabel('Hidden Neuron');
ylabel('Specificity');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Hardlim')

subplot(3,2,4)       % add fourth plot in 2 x 2 grid
plot(mean_radbasresult(:,1),mean_radbasresult(:,5))
xlabel('Hidden Neuron');
ylabel('Specificity');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Radial Basis')

subplot(3,2,5)       % add fourth plot in 2 x 2 grid
plot(mean_tribasresult(:,1),mean_tribasresult(:,5))
xlabel('Hidden Neuron');
ylabel('Specificity');
title('Tes Akurasi ELM Menggunakan Fungsi Aktivasi Tribas')


subplot(3,2,6)
plot(mean_sinresult(:,1),mean_sinresult(:,5))
hold on
plot(mean_sigresult(:,1),mean_sigresult(:,5))   
hold on
plot(mean_hardlimresult(:,1),mean_hardlimresult(:,5)) % line plot % stem plot
hold on
plot(mean_radbasresult(:,1),mean_radbasresult(:,5))
hold on
plot(mean_tribasresult(:,1),mean_tribasresult(:,5))
xlabel('Hidden Neuron');
ylabel('Accuracy');
l = legend('sin','sig','Hardlim','Radbas','Tribas');

