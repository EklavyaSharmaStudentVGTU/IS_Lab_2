% Step 1: Generate input and output data
X = rand(20, 1); % Generate 20 random input values between 0 and 1
Y = 1 + (0.4 * cos(2*pi*X/0.9)) + (0.5 * cos(2*pi*X))/2; % Calculate corresponding desired output values

% Step 2: Initialize the network parameters
N_i = 1; % Number of neurons in the input layer
N_h = 7; % Number of neurons in the hidden layer
N_o = 1; % Number of neurons in the output layer
learning_rate = 0.1; % Learning rate for weight updates
epsilon = 0.01; % Stopping criteria threshold
max_epochs = 1000; % Maximum number of training epochs

% Step 3: Initialize random weights and biases
W_i_h = randn(N_h, N_i); % Weight matrix for input to hidden layer
b_h = randn(N_h, 1); % Bias vector for hidden layer
W_h_o = randn(N_o, N_h); % Weight matrix for hidden to output layer
b_o = randn(N_o, 1); % Bias vector for output layer

epoch = 0;
while epoch < max_epochs

    % Step 4: Forward propagation

    H_i = W_i_h .* X' + b_h; % Hidden layer input
    H_Af = tanh(H_i); % Hidden layer hyperbolic tangent activation function
    O_i = W_h_o * H_Af + b_o; % Output layer input
    O_Af = O_i; % Output layer linear activation function
    
    % Step 5: Calculate loss/error
    error = O_Af - Y';
    
    % Step 6: Backpropagation

    delta2 = error; % Output layer delta
    dW_h_o = (1/size(X, 1)) * delta2 * H_Af'; % Weight gradient for hidden to output layer
    db_o = (1/size(X, 1)) * sum(delta2, 2); % Bias gradient for output layer
    
    delta1 = W_h_o' * delta2 .* (1 - H_Af.^2); % Hidden layer delta
    dW_i_h = (1/size(X, 1)) * delta1 .* X'; % Weight gradient for input to hidden layer
    db_h = (1/size(X, 1)) * sum(delta1, 2); % Bias gradient for hidden layer
    
    % Step 7: Update weights and biases

    W_i_h = W_i_h - learning_rate * dW_i_h;
    b_h = b_h - learning_rate * db_h;
    W_h_o = W_h_o - learning_rate * dW_h_o;
    b_o = b_o - learning_rate * db_o;
    
    % Step 8: Check stopping criteria

    if max(abs(error)) < epsilon
        break;
    end
    
    epoch = epoch + 1;
end

% Step 9: Evaluate the trained network

H_i = W_i_h .* X' + b_h;
H_Af = tanh(H_i);
O_i = W_h_o * H_Af + b_o;
O_Af = O_i;

% Plot the original and predicted outputs

plot(X, Y, 'b', X, O_Af, 'r');
legend('Desired Output', 'Predicted Output');
xlabel('Input');
ylabel('Output');

%% Surface Approximation

% Step 1: Generate the two inputs and one output data
S1 = rand(20, 1); % Generate first input having 20 random values between 0 and 1
S2 = rand(20, 1); % Generate first input having 20 random values between 0 and 1
Y1 = 1 + (0.4 * cos(2*pi*S1/0.9)) + (0.5 * cos(2*pi*S1))/2; % Calculate corresponding desired output values

% Combine the two input variables into one matrix
Z = [S1, S2];

% Rest of the code remains the same until Step 3

% Step 2: Initialize random weights and biases
neurons_input_layer = 2; % Number of neurons in the input layer (two input variables)
neurons_hidden_layer = 6; % Number of neurons in the hidden layer
neurons_output_layer = 1; % Number of neurons in the output layer
learning_rate1 = 0.1; % Learning rate for weight updates
epsilon1 = 0.01; % Stopping criteria threshold
max_epochs1 = 1000; % Maximum number of training epochs

% Step 3: Initialize random weights and biases
W1 = randn(neurons_hidden_layer, neurons_input_layer); % Weight matrix for input to hidden layer
b1 = randn(neurons_hidden_layer, 1); % Bias vector for hidden layer
W2 = randn(neurons_output_layer, neurons_hidden_layer); % Weight matrix for hidden to output layer
b2 = randn(neurons_output_layer, 1); % Bias vector for output layer


epoch1 = 0;
while epoch1 < max_epochs1

    % Step 4: Forward propagation

    HLI = W1 * Z' + b1; % Hidden layer input
    HLAF = tanh(HLI); % Hidden layer activation function
    OLI = W2 * HLAF + b2; % Output layer input
    OLAF = OLI; % Output layer activation function
    
    % Step 5: Calculate loss/error
    error1 = OLAF - Y1';

    % Step 6: Backpropagation (adjustments needed for two input variables)

    delta21 = error1; % Output layer delta
    dW2 = (1/size(Z, 1)) * delta21 * HLAF'; % Weight gradient for hidden to output layer
    db2 = (1/size(Z, 1)) * sum(delta21, 2); % Bias gradient for output layer

    delta1 = W2' * delta21 .* (1 - HLAF.^2); % Hidden layer delta
    dW1 = (1/size(Z, 1)) * delta1 * Z; % Weight gradient for input to hidden layer
    db1 = (1/size(Z, 1)) * sum(delta1, 2); % Bias gradient for hidden layer

% Step 7: Update weights and biases

    W1 = W1 - learning_rate1 * dW1;
    b1 = b1 - learning_rate1 * db1;
    W2 = W2 - learning_rate1 * dW2;
    b2 = b2 - learning_rate1 * db2;
% Rest of the code remains the same

% Step 8: Check stopping criteria

    if max(abs(error1)) < epsilon1
       break;
    end
    
    epoch1 = epoch1 + 1;
end

% Step 9: Evaluate the trained network

HLI = (W1 * Z' + b1);
HLAF = tanh(HLI);
OLI = W2 * HLAF + b2;
OLAF = OLI;

% Plot the original and predicted outputs
figure;
subplot(1, 2, 1);
scatter3(S1, S2, Y1, 'b', 'filled');
title('Desired Output');
xlabel('Input 1');
ylabel('Input 2');
zlabel('Output');

subplot(1, 2, 2);
scatter3(S1, S2, OLAF, 'r', 'filled');
title('Predicted Output');
xlabel('Input 1');
ylabel('Input 2');
zlabel('Output');
