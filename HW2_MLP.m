%==========================================================================
% Project 2
%==========================================================================

%% Loading

% Load the train and test images
training_images = loadMNISTImages('train-images.idx3-ubyte');
% train_images(:,i) is a double matrix of size 784xi(where i = 1 to 60000)
% intensity rescale to [0,1]

training_labels = loadMNISTLabels('train-labels.idx1-ubyte');
% train_labels(i) - 60000x1 vector

testing_images = loadMNISTImages('t10k-images.idx3-ubyte');
% testing_images(:,i) is a double matrix of size 784xi(where i = 1 to 10000)
% intensity rescale to [0,1]

testing_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% test_labels(i) - 10000x1 vector

%% Tasks 1, 2, and 3

% Prepare experinment data
number_of_training_images = 1000;
number_of_testing_images = 300;
[training_data, training_data_label] = balance_MNIST_selection(...
    training_images,training_labels,number_of_training_images);
[testing_data, testing_data_label] = balance_MNIST_selection(...
    testing_images,testing_labels,number_of_testing_images);

%-------------------------------------
% Training with the first learning rate
%-------------------------------------

Learning_rate_1_a = 0.01; % Learning_rate
Hidden_layer_1 = 10;
Momentum = 0.05;
input_neurons = 784;
output_neurons = 10;
Error_threshold = 0.001;

[Error_for_plot_1_a, V_1_a, W_1_a, Number_of_iterations_1_a] = MultiLayerPerceptron(training_data, training_data_label, input_neurons, Hidden_layer_1, output_neurons, Learning_rate_1_a, Momentum, Error_threshold);

figure(1);
plot(Error_for_plot_1_a)
grid;
str = sprintf('learning rate %g, %d hidden nodes,\n momentum term %g'...
    ,Learning_rate_1_a,Hidden_layer_1,Momentum);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0.5;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data,2)
 % from 1 to 1000

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data(:,m_test),[1,Hidden_layer_1]);

    % layer 1
        Net_test_k = sum(V_1_a.*test_x_i);     
        u_test_ki = 1./(1+ exp(-Net_test_k));   
        u_test_k = repmat(u_test_ki',[1,10]);

    % layer 2    
        Net_test_j = sum(W_1_a.*u_test_k);      
        y_t = 1./(1+ exp(-Net_test_j)); 

    % threshold
    y_test(y_t > Threshold) = 1;
    y_real(1,testing_data_label(m_test)+1) = 1;
    
    if y_test(testing_data_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(2);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_1,Learning_rate_1_a);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_1,Learning_rate_1_a);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the second learning rate
%-------------------------------------

Learning_rate_1_b = 0.1; % Learning_rate
Hidden_layer_1 = 10;
Momentum = 0.05;
input_neurons = 784;
output_neurons = 10;
Error_threshold = 0.001;

[Error_for_plot_1_b, V_1_b, W_1_b, Number_of_iterations_1_b] = MultiLayerPerceptron(training_data, training_data_label, input_neurons, Hidden_layer_1, output_neurons, Learning_rate_1_b, Momentum, Error_threshold);

figure(3);
plot(Error_for_plot_1_b)
grid;
str = sprintf('learning rate %g, %d hidden nodes,\n momentum term %g'...
    ,Learning_rate_1_b,Hidden_layer_1,Momentum);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0.5;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data,2)
 % from 1 to 1000

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data(:,m_test),[1,Hidden_layer_1]);

    % layer 1
        Net_test_k = sum(V_1_b.*test_x_i);     
        u_test_ki = 1./(1+ exp(-Net_test_k));   
        u_test_k = repmat(u_test_ki',[1,10]);

    % layer 2    
        Net_test_j = sum(W_1_b.*u_test_k);      
        y_t = 1./(1+ exp(-Net_test_j)); 

    % threshold
    y_test(y_t > Threshold) = 1;
    y_real(1,testing_data_label(m_test)+1) = 1;
    
    if y_test(testing_data_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(4);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_1,Learning_rate_1_b);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_1,Learning_rate_1_b);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the third learning rate
%-------------------------------------

Learning_rate_1_c = 0.5; % Learning_rate
Hidden_layer_1 = 10;
Momentum = 0.05;
input_neurons = 784;
output_neurons = 10;
Error_threshold = 0.001;

[Error_for_plot_1_c, V_1_c, W_1_c, Number_of_iterations_1_c] = MultiLayerPerceptron(training_data, training_data_label, input_neurons, Hidden_layer_1, output_neurons, Learning_rate_1_c, Momentum, Error_threshold);

figure(5);
plot(Error_for_plot_1_c)
grid;
str = sprintf('learning rate %g, %d hidden nodes,\n momentum term %g'...
    ,Learning_rate_1_c,Hidden_layer_1,Momentum);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0.5;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data,2)
 % from 1 to 1000

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data(:,m_test),[1,Hidden_layer_1]);

    % layer 1
        Net_test_k = sum(V_1_c.*test_x_i);     
        u_test_ki = 1./(1+ exp(-Net_test_k));   
        u_test_k = repmat(u_test_ki',[1,10]);

    % layer 2    
        Net_test_j = sum(W_1_c.*u_test_k);      
        y_t = 1./(1+ exp(-Net_test_j)); 

    % threshold
    y_test(y_t > Threshold) = 1;
    y_real(1,testing_data_label(m_test)+1) = 1;
    
    if y_test(testing_data_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(6);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_1,Learning_rate_1_c);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_1,Learning_rate_1_c);
title(str_test);
xlabel('digits');
ylabel('FPR');

%% Task 4

%-------------------------------------
% Training with the first learning rate
%-------------------------------------

Learning_rate_4_a = 0.01; % Learning_rate
Hidden_layer_4 = 20;
Momentum = 0.05;
input_neurons = 784;
output_neurons = 10;
Error_threshold = 0.001;

[Error_for_plot_4_a, V_4_a, W_4_a, Number_of_iterations_4_a] = MultiLayerPerceptron(training_data, training_data_label, input_neurons, Hidden_layer_4, output_neurons, Learning_rate_4_a, Momentum, Error_threshold);

figure(7);
plot(Error_for_plot_4_a)
grid;
str = sprintf('learning rate %g, %d hidden nodes,\n momentum term %g'...
    ,Learning_rate_4_a,Hidden_layer_4,Momentum);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0.5;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data,2)
 % from 1 to 1000

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data(:,m_test),[1,Hidden_layer_4]);

    % layer 1
        Net_test_k = sum(V_4_a.*test_x_i);     
        u_test_ki = 1./(1+ exp(-Net_test_k));   
        u_test_k = repmat(u_test_ki',[1,10]);

    % layer 2    
        Net_test_j = sum(W_4_a.*u_test_k);      
        y_t = 1./(1+ exp(-Net_test_j)); 

    % threshold
    y_test(y_t > Threshold) = 1;
    y_real(1,testing_data_label(m_test)+1) = 1;
    
    if y_test(testing_data_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(8);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_4,Learning_rate_4_a);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_4,Learning_rate_4_a);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the second learning rate
%-------------------------------------

Learning_rate_4_b = 0.1; % Learning_rate
Hidden_layer_4 = 20;
Momentum = 0.05;
input_neurons = 784;
output_neurons = 10;
Error_threshold = 0.001;

[Error_for_plot_4_b, V_4_b, W_4_b, Number_of_iterations_4_b] = MultiLayerPerceptron(training_data, training_data_label, input_neurons, Hidden_layer_4, output_neurons, Learning_rate_4_b, Momentum, Error_threshold);

figure(9);
plot(Error_for_plot_4_b)
grid;
str = sprintf('learning rate %g, %d hidden nodes,\n momentum term %g'...
    ,Learning_rate_4_b,Hidden_layer_4,Momentum);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0.5;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data,2)
 % from 1 to 1000

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data(:,m_test),[1,Hidden_layer_4]);

    % layer 1
        Net_test_k = sum(V_4_b.*test_x_i);     
        u_test_ki = 1./(1+ exp(-Net_test_k));   
        u_test_k = repmat(u_test_ki',[1,10]);

    % layer 2    
        Net_test_j = sum(W_4_b.*u_test_k);      
        y_t = 1./(1+ exp(-Net_test_j)); 

    % threshold
    y_test(y_t > Threshold) = 1;
    y_real(1,testing_data_label(m_test)+1) = 1;
    
    if y_test(testing_data_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(10);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_4,Learning_rate_4_b);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_4,Learning_rate_4_b);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the third learning rate
%-------------------------------------

Learning_rate_4_c = 0.5; % Learning_rate
Hidden_layer_4 = 20;
Momentum = 0.05;
input_neurons = 784;
output_neurons = 10;
Error_threshold = 0.001;

[Error_for_plot_4_c, V_4_c, W_4_c, Number_of_iterations_4_c] = MultiLayerPerceptron(training_data, training_data_label, input_neurons, Hidden_layer_4, output_neurons, Learning_rate_4_c, Momentum, Error_threshold);

figure(11);
plot(Error_for_plot_4_c)
grid;
str = sprintf('learning rate %g, %d hidden nodes,\n momentum term %g'...
    ,Learning_rate_4_c,Hidden_layer_4,Momentum);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0.5;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data,2)
 % from 1 to 1000

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data(:,m_test),[1,Hidden_layer_4]);

    % layer 1
        Net_test_k = sum(V_4_c.*test_x_i);     
        u_test_ki = 1./(1+ exp(-Net_test_k));   
        u_test_k = repmat(u_test_ki',[1,10]);

    % layer 2    
        Net_test_j = sum(W_4_c.*u_test_k);      
        y_t = 1./(1+ exp(-Net_test_j)); 

    % threshold
    y_test(y_t > Threshold) = 1;
    y_real(1,testing_data_label(m_test)+1) = 1;
    
    if y_test(testing_data_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(12);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_4,Learning_rate_4_c);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(training_data,2),size(testing_data,2),Hidden_layer_4,Learning_rate_4_c);
title(str_test);
xlabel('digits');
ylabel('FPR');
%==========================================================================