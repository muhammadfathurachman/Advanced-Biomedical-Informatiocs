function [result] = binaryVector(target)

[n m] = size(target);
hasil = zeros(n+1,m);

index_zeros = find(target == 1);
index_ones = find(target == 0);
hasil(1,index_zeros) = 1;
hasil(2,index_ones) = 1;
result = hasil;

end