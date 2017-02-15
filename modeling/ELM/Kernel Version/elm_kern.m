function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,Y,TY,OutputWeight] = elm_kern(Omega_train, train_labels, Omega_test, test_labels, Elm_Type, Regularization_coefficient, verbose)
% Furkan :
% [~, ~, ~, TestingAccuracy,Y,TY,~] = elm_kern(Omega_train, train_labels, Omega_test, test_labels, Elm_Type, Regularization_coefficient)
% kernels should be [N_Test x N_Train]
if size(Omega_train,2) ~= size(Omega_test,2)
    error('kernels should be [N_Test x N_Train].');
end
if size(Omega_train,1) ~= numel(train_labels) || size(Omega_test,1) ~= numel(test_labels)
    error('Kernel size is not consistent with the num. of labels.');
end

if nargin<7
    verbose=0;
elseif verbose==1
    fprintf('%s.m: ',mfilename);
    %disp('verbose');
    if Elm_Type==0
        %fprintf('Labels btw. ~(%d-%d), %d Train, %d Test samples\n',round(min(train_labels)),round(max(train_labels)) , size(Omega_train,1), size(Omega_test,1));
        %fprintf('Labels btw. (%s-%s), %d Train, %d Test samples\n',num2str(min(train_labels)),num2str(max(train_labels)) , size(Omega_train,1), size(Omega_test,1));
        fprintf('fold 1: %d Train, %d Test samples\n',size(Omega_train,1), size(Omega_test,1));
    else
        fprintf('%d cls., %d train, %d test.\n',numel(unique(train_labels)) , size(Omega_train,1), size(Omega_test,1));
        % also if binary classification, give class distributions:
        if numel(unique(train_labels)) == 2
            utl = unique(train_labels); % unique_train_labels
            fprintf('label=%d/%d: train= [%d/%d], test=[%d/%d]\n',utl(1),utl(2), ...
                sum(train_labels==utl(1)),sum(train_labels==utl(2)),sum(test_labels==utl(1)),sum(test_labels==utl(2)));
        end
    end
end

% Usage: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,Y,TY,OutputWeight,Omega_train,Omega_test] = elm_kern 
% (Omega_train, train_labels, Omega_test, test_labels, Elm_Type, Regularization_coefficient)

%
% Input:
% TrainingData_File           - Filename of training data set
% TestingData_File            - Filename of testing data set
% Elm_Type                    - 0 for regression; 1 for (both binary and multi-classes) classification
% Regularization_coefficient  - Regularization coefficient C
% Kernel_type                 - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%Kernel_para                  - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...
% Output: 
% TrainingTime                - Time (seconds) spent on training ELM
% TestingTime                 - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy            - Training accuracy: 
%                               RMSE for regression or correct classification rate for classification
% TestingAccuracy             - Testing accuracy: 
%                               RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm_kernel('sinc_train', 'sinc_test', 0, 1, ''RBF_kernel',100)
% Sample2 classification: elm_kernel('diabetes_train', 'diabetes_test', 1, 1, 'RBF_kernel',100)
%
    %%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       MARCH 2012

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

% Reshape input:
train_labels = reshape(train_labels,[],1);
test_labels = reshape(test_labels,[],1);

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
T=train_labels'; %train_data(1,:);%';
%P=train_data(2:size(train_data,1),:);%';
clear train_labels;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
%test_data=load(TestingData_File);
TV.T= test_labels';
%test_data(1,:);%';
%TV.P=test_data(2:size(test_data,1),:);%';
clear test_labels;                                    %   Release raw testing data array

C = Regularization_coefficient;
NumberofTrainingData=size(T,2);
NumberofTestingData=size(TV.T,2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
                                              %   end if of Elm_Type
end

%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
n = size(T,2);
%OutputWeight=((Omega_train+speye(n)/C)\(T')); 
%Furkan_sil = (Omega_train+eye(n)/C);
OutputWeight=((Omega_train+eye(n)/C)\(T')); 
TrainingTime=toc;

%%%%%%%%%%% Calculate the training output
Y=(Omega_train * OutputWeight)';                             %   Y: the actual output of the training data

%%%%%%%%%%% Calculate the output of testing input
tic;
TY=(Omega_test * OutputWeight)';                            %   TY: the actual output of the testing data
TestingTime=toc;

%%%%%%%%%% Calculate training & testing classification accuracy

if Elm_Type == REGRESSION
%%%%%%%%%% Calculate training & testing accuracy (RMSE) for regression case
    TrainingAccuracy=sqrt(mse(T - Y));
    TestingAccuracy=sqrt(mse(TV.T - TY)) ;          
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);  
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2) ; 
end
    
    
