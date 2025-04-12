
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <time.h>
 
 #define LEARNING_RATE 0.01
 #define EPOCHS 20  // Reduced for faster testing
 #define BATCH_SIZE 64
 #define EPSILON 1e-6
 #define TRAINING_SAMPLES 1000  // Reduced for faster testing
 #define INPUT_SIZE 100  // Reduced for faster testing
 #define HIDDEN_SIZE 50
 #define OUTPUT_SIZE 10
 
 // Matrix structure
 typedef struct {
     int rows;
     int cols;
     double* data;  // Row-major order for better cache locality
 } Matrix;
 
 // Neural network layer
 typedef struct {
     Matrix weights;
     Matrix bias;
     Matrix z;       // Pre-activation values
     Matrix a;       // Activation values
     Matrix delta;   // Error delta
     Matrix dw;      // Weight gradients
     Matrix db;      // Bias gradients
 } Layer;
 
 // Neural network
 typedef struct {
     int num_layers;
     Layer* layers;
 } NeuralNetwork;
 
 // Create a matrix with specified dimensions
 Matrix create_matrix(int rows, int cols) {
     Matrix m;
     m.rows = rows;
     m.cols = cols;
     m.data = (double*)calloc(rows * cols, sizeof(double));
     return m;
 }
 
 // Free matrix memory
 void free_matrix(Matrix* m) {
     free(m->data);
     m->data = NULL;
     m->rows = 0;
     m->cols = 0;
 }
 
 // Initialize matrix with random values
 void init_matrix_random(Matrix* m, double scale) {
     for (int i = 0; i < m->rows * m->cols; i++) {
         m->data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
     }
 }
 
 // Set all matrix elements to zero
 void zero_matrix(Matrix* m) {
     memset(m->data, 0, m->rows * m->cols * sizeof(double));
 }
 
 // Matrix multiplication: C = A * B
 void matrix_multiply(Matrix* A, Matrix* B, Matrix* C) {
     if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication (%dx%d * %dx%d -> %dx%d)\n", 
                 A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
         exit(1);
     }
     
     zero_matrix(C);
     
     for (int i = 0; i < A->rows; i++) {
         for (int k = 0; k < A->cols; k++) {
             double A_ik = A->data[i * A->cols + k];
             for (int j = 0; j < B->cols; j++) {
                 C->data[i * C->cols + j] += A_ik * B->data[k * B->cols + j];
             }
         }
     }
 }
 
 // Matrix addition: C = A + B
 void matrix_add(Matrix* A, Matrix* B, Matrix* C) {
     if (A->rows != B->rows || A->cols != B->cols || 
         C->rows != A->rows || C->cols != A->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for addition\n");
         exit(1);
     }
     
     for (int i = 0; i < A->rows * A->cols; i++) {
         C->data[i] = A->data[i] + B->data[i];
     }
 }
 
 // Matrix transpose: B = A^T
 void matrix_transpose(Matrix* A, Matrix* B) {
     if (A->rows != B->cols || A->cols != B->rows) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for transpose\n");
         exit(1);
     }
     
     for (int i = 0; i < A->rows; i++) {
         for (int j = 0; j < A->cols; j++) {
             B->data[j * B->cols + i] = A->data[i * A->cols + j];
         }
     }
 }
 
 // Element-wise sigmoid activation function
 void sigmoid(Matrix* m, Matrix* output) {
     if (m->rows != output->rows || m->cols != output->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for sigmoid\n");
         exit(1);
     }
     
     for (int i = 0; i < m->rows * m->cols; i++) {
         output->data[i] = 1.0 / (1.0 + exp(-m->data[i]));
     }
 }
 
 // Element-wise sigmoid derivative
 void sigmoid_derivative(Matrix* a, Matrix* output) {
     if (a->rows != output->rows || a->cols != output->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for sigmoid derivative\n");
         exit(1);
     }
     
     for (int i = 0; i < a->rows * a->cols; i++) {
         output->data[i] = a->data[i] * (1.0 - a->data[i]);
     }
 }
 
 // Element-wise ReLU activation function
 void relu(Matrix* m, Matrix* output) {
     if (m->rows != output->rows || m->cols != output->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for ReLU\n");
         exit(1);
     }
     
     for (int i = 0; i < m->rows * m->cols; i++) {
         output->data[i] = m->data[i] > 0 ? m->data[i] : 0;
     }
 }
 
 // Element-wise ReLU derivative
 void relu_derivative(Matrix* z, Matrix* output) {
     if (z->rows != output->rows || z->cols != output->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for ReLU derivative\n");
         exit(1);
     }
     
     for (int i = 0; i < z->rows * z->cols; i++) {
         output->data[i] = z->data[i] > 0 ? 1.0 : 0.0;
     }
 }
 
 // Element-wise matrix multiplication (Hadamard product): C = A ⊙ B
 void matrix_multiply_elementwise(Matrix* A, Matrix* B, Matrix* C) {
     if (A->rows != B->rows || A->cols != B->cols || 
         C->rows != A->rows || C->cols != A->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for element-wise multiplication\n");
         exit(1);
     }
     
     for (int i = 0; i < A->rows * A->cols; i++) {
         C->data[i] = A->data[i] * B->data[i];
     }
 }
 
 // Matrix subtraction: C = A - B
 void matrix_subtract(Matrix* A, Matrix* B, Matrix* C) {
     if (A->rows != B->rows || A->cols != B->cols || 
         C->rows != A->rows || C->cols != A->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for subtraction\n");
         exit(1);
     }
     
     for (int i = 0; i < A->rows * A->cols; i++) {
         C->data[i] = A->data[i] - B->data[i];
     }
 }
 
 // Scale matrix: B = A * scalar
 void matrix_scale(Matrix* A, double scalar, Matrix* B) {
     if (A->rows != B->rows || A->cols != B->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for scaling\n");
         exit(1);
     }
     
     for (int i = 0; i < A->rows * A->cols; i++) {
         B->data[i] = A->data[i] * scalar;
     }
 }
 
 // Create a neural network with specified layer sizes
 NeuralNetwork create_neural_network(int num_layers, int* layer_sizes) {
     NeuralNetwork nn;
     nn.num_layers = num_layers - 1;  // Number of weight layers (excluding input layer)
     nn.layers = (Layer*)malloc(nn.num_layers * sizeof(Layer));
     
     for (int i = 0; i < nn.num_layers; i++) {
         int input_size = layer_sizes[i];
         int output_size = layer_sizes[i + 1];
         
         // Initialize weights and biases
         nn.layers[i].weights = create_matrix(input_size, output_size);
         nn.layers[i].bias = create_matrix(1, output_size);
         
         // Initialize with He initialization
         double scale = sqrt(2.0 / input_size);
         init_matrix_random(&nn.layers[i].weights, scale);
         init_matrix_random(&nn.layers[i].bias, 0.1);
         
         // Create matrices for forward and backward pass
         nn.layers[i].z = create_matrix(BATCH_SIZE, output_size);
         nn.layers[i].a = create_matrix(BATCH_SIZE, output_size);
         nn.layers[i].delta = create_matrix(BATCH_SIZE, output_size);
         nn.layers[i].dw = create_matrix(input_size, output_size);
         nn.layers[i].db = create_matrix(1, output_size);
     }
     
     return nn;
 }
 
 // Free neural network memory
 void free_neural_network(NeuralNetwork* nn) {
     for (int i = 0; i < nn->num_layers; i++) {
         free_matrix(&nn->layers[i].weights);
         free_matrix(&nn->layers[i].bias);
         free_matrix(&nn->layers[i].z);
         free_matrix(&nn->layers[i].a);
         free_matrix(&nn->layers[i].delta);
         free_matrix(&nn->layers[i].dw);
         free_matrix(&nn->layers[i].db);
     }
     free(nn->layers);
     nn->layers = NULL;
     nn->num_layers = 0;
 }
 
 // Forward pass through the network
 void forward_pass(NeuralNetwork* nn, Matrix* input) {
     // Input to first hidden layer
     matrix_multiply(input, &nn->layers[0].weights, &nn->layers[0].z);
     
     // Add bias to each row
     for (int i = 0; i < input->rows; i++) {
         for (int j = 0; j < nn->layers[0].bias.cols; j++) {
             nn->layers[0].z.data[i * nn->layers[0].z.cols + j] += nn->layers[0].bias.data[j];
         }
     }
     
     // Apply activation function
     relu(&nn->layers[0].z, &nn->layers[0].a);
     
     // Hidden layers to output
     for (int i = 1; i < nn->num_layers; i++) {
         matrix_multiply(&nn->layers[i-1].a, &nn->layers[i].weights, &nn->layers[i].z);
         
         // Add bias to each row
         for (int j = 0; j < nn->layers[i-1].a.rows; j++) {
             for (int k = 0; k < nn->layers[i].bias.cols; k++) {
                 nn->layers[i].z.data[j * nn->layers[i].z.cols + k] += nn->layers[i].bias.data[k];
             }
         }
         
         // Apply activation function (sigmoid for output layer, ReLU for hidden layers)
         if (i == nn->num_layers - 1) {
             sigmoid(&nn->layers[i].z, &nn->layers[i].a);
         } else {
             relu(&nn->layers[i].z, &nn->layers[i].a);
         }
     }
 }
 
 // Backward pass through the network
 void backward_pass(NeuralNetwork* nn, Matrix* input, Matrix* target) {
     int output_layer = nn->num_layers - 1;
     
     // Output layer error
     matrix_subtract(&nn->layers[output_layer].a, target, &nn->layers[output_layer].delta);
     
     // Backpropagate error
     for (int i = output_layer; i > 0; i--) {
         // Calculate gradients
         Matrix a_transpose = create_matrix(nn->layers[i-1].a.cols, nn->layers[i-1].a.rows);
         matrix_transpose(&nn->layers[i-1].a, &a_transpose);
         matrix_multiply(&a_transpose, &nn->layers[i].delta, &nn->layers[i].dw);
         free_matrix(&a_transpose);
         
         // Calculate bias gradients (sum over batch)
         zero_matrix(&nn->layers[i].db);
         for (int j = 0; j < nn->layers[i].delta.rows; j++) {
             for (int k = 0; k < nn->layers[i].delta.cols; k++) {
                 nn->layers[i].db.data[k] += nn->layers[i].delta.data[j * nn->layers[i].delta.cols + k];
             }
         }
         
         // Propagate error to previous layer
         if (i > 0) {
             Matrix weights_transpose = create_matrix(nn->layers[i].weights.cols, nn->layers[i].weights.rows);
             matrix_transpose(&nn->layers[i].weights, &weights_transpose);
             
             Matrix delta_prev = create_matrix(nn->layers[i].delta.rows, nn->layers[i].weights.rows);
             matrix_multiply(&nn->layers[i].delta, &weights_transpose, &delta_prev);
             
             Matrix activation_derivative = create_matrix(nn->layers[i-1].z.rows, nn->layers[i-1].z.cols);
             relu_derivative(&nn->layers[i-1].z, &activation_derivative);
             
             matrix_multiply_elementwise(&delta_prev, &activation_derivative, &nn->layers[i-1].delta);
             
             free_matrix(&weights_transpose);
             free_matrix(&delta_prev);
             free_matrix(&activation_derivative);
         }
     }
     
     // First layer gradients
     Matrix input_transpose = create_matrix(input->cols, input->rows);
     matrix_transpose(input, &input_transpose);
     matrix_multiply(&input_transpose, &nn->layers[0].delta, &nn->layers[0].dw);
     free_matrix(&input_transpose);
     
     // First layer bias gradients
     zero_matrix(&nn->layers[0].db);
     for (int j = 0; j < nn->layers[0].delta.rows; j++) {
         for (int k = 0; k < nn->layers[0].delta.cols; k++) {
             nn->layers[0].db.data[k] += nn->layers[0].delta.data[j * nn->layers[0].delta.cols + k];
         }
     }
 }
 
 // Update weights and biases
 void update_weights(NeuralNetwork* nn, double learning_rate, int batch_size) {
     double lr_batch = learning_rate / batch_size;
     
     for (int i = 0; i < nn->num_layers; i++) {
         // Update weights
         for (int j = 0; j < nn->layers[i].weights.rows * nn->layers[i].weights.cols; j++) {
             nn->layers[i].weights.data[j] -= lr_batch * nn->layers[i].dw.data[j];
         }
         
         // Update biases
         for (int j = 0; j < nn->layers[i].bias.cols; j++) {
             nn->layers[i].bias.data[j] -= lr_batch * nn->layers[i].db.data[j];
         }
     }
 }
 
 // Calculate mean squared error
 double calculate_mse(Matrix* output, Matrix* target) {
     if (output->rows != target->rows || output->cols != target->cols) {
         fprintf(stderr, "Error: Incompatible matrix dimensions for MSE calculation\n");
         exit(1);
     }
     
     double sum_error = 0.0;
     for (int i = 0; i < output->rows * output->cols; i++) {
         double error = output->data[i] - target->data[i];
         sum_error += error * error;
     }
     
     return sum_error / (output->rows * output->cols);
 }
 
 // Train the neural network
 void train(NeuralNetwork* nn, Matrix* inputs, Matrix* targets, int num_samples, int epochs, int batch_size, double learning_rate) {
     int num_batches = (num_samples + batch_size - 1) / batch_size;
     
     // Temporary matrices for batch processing
     Matrix batch_input = create_matrix(batch_size, inputs->cols);
     Matrix batch_target = create_matrix(batch_size, targets->cols);
     
     for (int epoch = 0; epoch < epochs; epoch++) {
         double total_error = 0.0;
         
         for (int batch = 0; batch < num_batches; batch++) {
             int start_idx = batch * batch_size;
             int end_idx = (batch + 1) * batch_size;
             if (end_idx > num_samples) end_idx = num_samples;
             int current_batch_size = end_idx - start_idx;
             
             // Copy batch data
             for (int i = 0; i < current_batch_size; i++) {
                 for (int j = 0; j < inputs->cols; j++) {
                     batch_input.data[i * inputs->cols + j] = inputs->data[(start_idx + i) * inputs->cols + j];
                 }
                 for (int j = 0; j < targets->cols; j++) {
                     batch_target.data[i * targets->cols + j] = targets->data[(start_idx + i) * targets->cols + j];
                 }
             }
             
             // Zero out unused samples in the batch
             if (current_batch_size < batch_size) {
                 for (int i = current_batch_size; i < batch_size; i++) {
                     for (int j = 0; j < inputs->cols; j++) {
                         batch_input.data[i * inputs->cols + j] = 0.0;
                     }
                     for (int j = 0; j < targets->cols; j++) {
                         batch_target.data[i * targets->cols + j] = 0.0;
                     }
                 }
             }
             
             // Forward pass
             forward_pass(nn, &batch_input);
             
             // Calculate error
             double batch_error = calculate_mse(&nn->layers[nn->num_layers-1].a, &batch_target);
             total_error += batch_error * current_batch_size;
             
             // Backward pass
             backward_pass(nn, &batch_input, &batch_target);
             
             // Update weights
             update_weights(nn, learning_rate, current_batch_size);
         }
         
         // Calculate average error
         double avg_error = total_error / num_samples;
         
         // Print progress every 5 epochs
         if (epoch % 5 == 0 || epoch == epochs - 1) {
             printf("Epoch %d/%d, MSE: %f\n", epoch + 1, epochs, avg_error);
         }
         
         // Early stopping
         if (avg_error < EPSILON) {
             printf("Converged at epoch %d with MSE: %f\n", epoch + 1, avg_error);
             break;
         }
     }
     
     free_matrix(&batch_input);
     free_matrix(&batch_target);
 }
 
 // Predict using the trained network
 void predict_single(NeuralNetwork* nn, double* input, double* output) {
     // Create temporary matrices for a single sample
     Matrix input_matrix = create_matrix(1, INPUT_SIZE);
     Matrix output_matrix = create_matrix(1, OUTPUT_SIZE);
     
     // Copy input data
     for (int i = 0; i < INPUT_SIZE; i++) {
         input_matrix.data[i] = input[i];
     }
     
     // Forward pass
     // First layer
     Matrix z1 = create_matrix(1, nn->layers[0].weights.cols);
     Matrix a1 = create_matrix(1, nn->layers[0].weights.cols);
     
     matrix_multiply(&input_matrix, &nn->layers[0].weights, &z1);
     
     // Add bias
     for (int j = 0; j < nn->layers[0].bias.cols; j++) {
         z1.data[j] += nn->layers[0].bias.data[j];
     }
     
     // Apply ReLU
     relu(&z1, &a1);
     
     // Second layer (output)
     Matrix z2 = create_matrix(1, nn->layers[1].weights.cols);
     Matrix a2 = create_matrix(1, nn->layers[1].weights.cols);
     
     matrix_multiply(&a1, &nn->layers[1].weights, &z2);
     
     // Add bias
     for (int j = 0; j < nn->layers[1].bias.cols; j++) {
         z2.data[j] += nn->layers[1].bias.data[j];
     }
     
     // Apply sigmoid for output layer
     sigmoid(&z2, &a2);
     
     // Copy output
     for (int i = 0; i < OUTPUT_SIZE; i++) {
         output[i] = a2.data[i];
     }
     
     // Free temporary matrices
     free_matrix(&input_matrix);
     free_matrix(&output_matrix);
     free_matrix(&z1);
     free_matrix(&a1);
     free_matrix(&z2);
     free_matrix(&a2);
 }
 
 // Create a random dataset for benchmarking
 void create_random_dataset(Matrix* inputs, Matrix* targets, int num_samples) {
     for (int i = 0; i < num_samples; i++) {
         // Generate random input values
         for (int j = 0; j < inputs->cols; j++) {
             inputs->data[i * inputs->cols + j] = ((double)rand() / RAND_MAX);
         }
         
         // Generate random target values (one-hot encoding)
         int target_class = rand() % targets->cols;
         for (int j = 0; j < targets->cols; j++) {
             targets->data[i * targets->cols + j] = (j == target_class) ? 1.0 : 0.0;
         }
     }
 }
 
 int main() {
     // Seed random number generator
     srand(time(NULL));
     
     clock_t start_time, end_time;
     double cpu_time_used;
     printf(" Neural Network with Matrix Multiplication\n");
     printf("--------------------------------------------------\n");
     printf("Training samples: %d\n", TRAINING_SAMPLES);
     printf("Network architecture: %d-%d-%d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
     
     // Define network architecture
     int num_layers = 3;
     int layer_sizes[] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
     
     // Create neural network
     NeuralNetwork nn = create_neural_network(num_layers, layer_sizes);
     
     // Create dataset
     Matrix inputs = create_matrix(TRAINING_SAMPLES, INPUT_SIZE);
     Matrix targets = create_matrix(TRAINING_SAMPLES, OUTPUT_SIZE);
     create_random_dataset(&inputs, &targets, TRAINING_SAMPLES);
     
     printf("\nTraining neural network...\n");
     
     start_time = clock();
     
     // Train the network
     train(&nn, &inputs, &targets, TRAINING_SAMPLES, EPOCHS, BATCH_SIZE, LEARNING_RATE);
     
     
     // Test the trained network on a few samples
     printf("\nTesting the trained network on 5 samples:\n");
     double output[OUTPUT_SIZE];

     for (int i = 0; i < 5 && i < TRAINING_SAMPLES; i++) {
         double single_input[INPUT_SIZE];
         for (int j = 0; j < INPUT_SIZE; j++) {
             single_input[j] = inputs.data[i * INPUT_SIZE + j];
         }
         
         predict_single(&nn, single_input, output);
         
         printf("Sample %d:\n", i+1);
         
         // Find the predicted and target classes
         int predicted_class = 0;
         int target_class = 0;
         double max_output = output[0];
         double max_target = targets.data[i * OUTPUT_SIZE];
         
         for (int j = 1; j < OUTPUT_SIZE; j++) {
             if (output[j] > max_output) {
                 max_output = output[j];
                 predicted_class = j;
             }
             if (targets.data[i * OUTPUT_SIZE + j] > max_target) {
                 max_target = targets.data[i * OUTPUT_SIZE + j];
                 target_class = j;
             }
         }
         
         printf("  Predicted class: %d, Target class: %d\n", predicted_class, target_class);
     }
     end_time = clock();
     cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
     printf("\n Testing completed in %f seconds\n", cpu_time_used);
     // Free memory
     free_matrix(&inputs);
     free_matrix(&targets);
     free_neural_network(&nn);
     
     return 0;
 }