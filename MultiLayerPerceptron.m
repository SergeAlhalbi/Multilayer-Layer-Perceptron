%/*******************************************************
% * Copyright (C) 2019-2020 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Artificial Neural Network.
% * 
% * MIT License
% *******************************************************/
%% Multi Layer Perceptron
function [Error_for_plot, v_ki, w_jk, Number_of_iterations] = MultiLayerPerceptron(input_data, labels, input_neurons, hidden_nodes, output_neurons, learning_rate, momentum, Error_threshold)
    % Input:  input_data            784 x number of training data
    %        labels                number of training data  x 1
    %        input_neurons         784
    %        hidden nodes          middle layer neuron numbers (H)
    %        output_neurons        10
    %        learning_rate         0.5, 0.1, 0.01
    %        momentum              momentum term rate 0.05, 0.1, 0.2
    %        Error_threshold       0.001
    
    % Output: Error_for_plot        MSE for each iteration
    %        v_ki, w_jk            The weights of this network
    
    H = hidden_nodes;
    v_ki = -0.5+rand(input_neurons,H);
    w_jk = -0.5+rand(H,output_neurons);
    delta_v_t = -0.5+rand(784,H);
    delta_w_t = -0.5+rand(H,10);
    iterations = 1;
    Error_end = 1;
    max_iterations = 1000;
    Error_update = zeros(max_iterations,1);

    while Error_end > Error_threshold && iterations < max_iterations
        E = 0;
        for m = 1 : size(input_data,2)
            % from 1 to 1000

            d = zeros(1,10);

            x_i = repmat(input_data(:,m),[1,H]);
            % size: 784 * Hidden nodes (H)

            Net_k = sum(v_ki.*x_i);    % step 1
            u_k = 1./(1+ exp(-Net_k)); % step 2
            % 1 * H

            u_k_x = repmat(u_k',[1,output_neurons]);
            % H * 10
            u_k_y = repmat(u_k,[input_neurons,1]);
            % 784 * H

            Net_j = sum(w_jk.*u_k_x);    % step 3
            y = 1./(1+ exp(-Net_j));     % step 4
            % 1 * 10

            y_j = repmat(y,[H,1]);   
            % H * 10

            d (labels(m)+1) = 1;

            sigma = d-y;             % step 5
            % 1 * 10
            sigma_j = repmat(sigma,[H,1]);
            % H * 10

            part_w1 = sigma_j.*u_k_x;         % step 6
            part_w2 = y_j.*(1 - y_j);
            delta_w = learning_rate*part_w1.*part_w2;  
            % H * 10

            sigma_k = sum(w_jk.*sigma_j.*y_j.*(1-y_j),2)';   % step 7
            % 1 * H
            sigma_kh = repmat(sigma_k,[784,1]);
            % 784 * H

            part_v1 = sigma_kh.*x_i;         % step 8
            part_v2 = u_k_y.*(1-u_k_y);
            delta_v = learning_rate*part_v1.*part_v2; 
            % 784 * H

            % using momentum term
            w_jk = w_jk + (1-momentum) * delta_w + momentum * delta_w_t;      %  step 9
            v_ki = v_ki + (1-momentum) * delta_v + momentum * delta_v_t;      %  step 10

            En = sum((d-y).^2);
            E = E + En;

            delta_w_t = delta_w;
            delta_v_t = delta_v;
        end

        Error_end = E/(10*size(input_data,2));
        Error_update(iterations) = Error_end;
        iterations = iterations+1;
    end
    Number_of_iterations = iterations-1;
    Error_for_plot = Error_update(1:iterations-1);
end
