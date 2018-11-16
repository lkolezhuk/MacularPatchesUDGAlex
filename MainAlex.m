clear all
close all
clc


%% Loading the training datasets

% TrainNeg = dir('Retinal Vessels Train\cropped\cropped\neg');
% TrainNeg = TrainNeg(~ismember({TrainNeg.name},{'.','..'}));


Training = imageDatastore('C:\Users\benja\Documents\Deep Learning Meriaudeau\Retinal Vessels Train\cropped\cropped\Train','IncludeSubfolders', true, 'LabelSource', 'foldernames');

alex = alexnet;
layers = alex.Layers;

layers(23) = fullyConnectedLayer(2);
layers(25) = classificationLayer;

% opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64);
% trainingImages.ReadFcn = @readFunctionTrain;
% 
% myNet = trainNetwork(trainingImages, layers, opts);
