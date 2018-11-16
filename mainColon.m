clear all;
close all;
clc;

load datasource_colon-2.mat;

train = colon_train./max(max(colon_train));
train_class = colon_tnCL - 1;
test = colon_test./max(max(colon_test));
test_class = colon_ttCL - 1;

p=1;
n_input = length(train);
n_middle = 10;
n_output = 1;

epoch = 1000; % number of runs during the learning stage
eta = 0.5; % learning rate

w1 = randn(n_middle, n_input)/sqrt(n_middle * p);
w2 = randn(n_output, n_middle)/sqrt(n_input * p);

b1 = zeros(n_middle, 1);
b2 = zeros(n_output, 1);

for k=1:epoch
    I(k) = 0; 
    ind = randperm(size(train,2));
    P = train;
    
    for i=1:size(train,2)
        v1 = w1 * P(:, ind(i)) + b1;
        y1 = tansig(v1);
        v2 = w2*y1+b2;
        y2 =  logsig(v2);
        
        out(i) = y2;
        e = train_class(ind(i)) - y2; % what is the diff between output and true value
        I(k) = I(k) + e*e'; % square error
        
        delta_2 = dlogsig(v2,y2) * e;
        delta_1 = gmultiply(dtansig(v1, y1), w2' * delta_2);
        
        w2 = w2 + eta * delta_2 * y1';
        w1 = w1 + eta * delta_1 * P(:, ind(i))';
        
        b2 = b2 + eta * delta_2;
        b1 = b1 + eta * delta_1;
                
    end
    
end

%%
out_1 = zeros(size(test,2),2);
score = 0;
for i = 1:size(test,2)
        v1 = w1*test(:,i) + b1;
        y1 = tansig(v1);
        v2 = w2*y1+b2;
        y2 =  logsig(v2);
        if y2>0.5
            y2=1;
        else
            y2=0;
        end
        out_1(i,:) = [y2 test_class(i)];
        if y2 == test_class(i);
            score = score + 1;
        end
end
out_1'
score/size(test,2)