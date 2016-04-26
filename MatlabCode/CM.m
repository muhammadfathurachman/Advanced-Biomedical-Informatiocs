function [Accuracy, Sensitivity, Specificity] = CM(Y_actual, Y_predicted)
TP = 0;
TN = 0;
FP = 0;
FN = 0;

[r,n] = size(Y_actual);

for i= 1:r
    if Y_actual(i,1) == 1 && Y_predicted(i,1) == 1
        TP = TP + 1;
    end
    if Y_actual(i,1) == 1 && Y_predicted(i,1) == 0
        FN = FN + 1;
    end
    if Y_actual(i,1) == 0 && Y_predicted(i,1) == 1
        FP = FP + 1;
    end
    if Y_actual(i,1) == 0 && Y_predicted(i,1) == 0
        TN = TN + 1;
    end
end

Accuracy =  (TP+TN)/(TN+TP+FP+FN);
Sensitivity = (TP)/(TP+FN);
Specificity = (TN)/ (TN+FP);

