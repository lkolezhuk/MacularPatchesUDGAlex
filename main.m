clear all;
close all;
clc;

x = [0 0 1 1];
y = [0 1 0 1];

f = [0 1 1 0];
P = [x' y'];

n_input = 2;
n_middle = 2;
n_output = 1;

epoch = 10000; % number of runs during the learning stage
eta = 0.5; % learning rate


w1 = randn(n_middle, n_input);
w2 = randn(n_output, n_middle);

b1 = zeros(n_middle, 1);
b2 = zeros(n_output, 1);

for k=1:epoch
    I(k) = 0; 
    ind = randperm(length(P));
    
    for i=1:length(P)
        v1 = w1*P(ind(i),:)' + b1;
        y1 = tansig(v1);
        v2 = w2*y1+b2;
        y2 =  logsig(v2);
        
        out(i) = y2;
        e = f(ind(i)) - y2; % what is the diff between output and true value
        I(k) = I(k) + e*e'; % square error
        
        delta_2 = dlogsig(v2,y2) * e;
        delta_1 = gmultiply(dtansig(v1, y1), w2' * delta_2);
        
        w2 = w2 + eta * delta_2 * y1';
        w1 = w1 + eta * delta_1 * P(ind(i),:);
        
        b2 = b2 + eta * delta_2;
        b1 = b1 + eta * delta_1;
                
    end
    
end


for i = 1:length(P)
        v1 = w1*P(i,:)' + b1;
        y1 = tansig(v1);
        v2 = w2*y1+b2;
        y2 =  logsig(v2);
        if y2>0.5
            y2=1;
        else
            y2=0;
        end
        out_1(i) = y2;
end
out_1'
