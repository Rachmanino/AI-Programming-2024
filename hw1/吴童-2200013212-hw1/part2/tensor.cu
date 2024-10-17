#include<stdio.h>
#include<assert.h>
#include<stdlib.h>
#include<vector> // for shape
#include<cuda_runtime.h>
#include<math.h>
//#include<pybind11/pybind11.h>

typedef std::vector<int> shape_t;

enum device_t {
    CPU,
    GPU
};


/* Begin Tensor definition */
class Tensor {
    public:
        static int TensorCount;
        Tensor(shape_t shape, device_t device);
        Tensor(const Tensor& tensor);
        ~Tensor(); // deep copy
        Tensor& cpu();
        Tensor& gpu();
    
        shape_t shape;
        device_t device;
        float* data; // Currently only support float32 and contiguous memory storage
        size_t size;
};
int Tensor::TensorCount = 0;

Tensor::Tensor(shape_t shape, device_t device) {
    assert (shape.size() > 0);
    assert (device == CPU || device == GPU);

    this->shape = shape;
    this->device = device;
    
    this->size = 1;
    for (int i = 0; i < shape.size(); i++) {
        this->size *= shape[i];
    }

    if (device == CPU) {
        this->data = (float*)malloc(this->size * sizeof(float));
    } else {
        cudaMalloc(&this->data, this->size * sizeof(float));
    }

    Tensor::TensorCount++;
}

/* We simply use deep copy here to avoid memory issues. */
Tensor::Tensor(const Tensor &tensor) {
    this->shape = tensor.shape;
    this->device = tensor.device;
    this->size = tensor.size;

    if (this->device == CPU) {
        this->data = (float*)malloc(this->size * sizeof(float));
        memcpy(this->data, tensor.data, this->size * sizeof(float));
    } else {
        cudaMalloc(&this->data, this->size * sizeof(float));
        cudaMemcpy(this->data, tensor.data, this->size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor::TensorCount++;
}

Tensor::~Tensor() {
    if (this->device == CPU) {
        free(this->data);
    } else {
        cudaFree(this->data);
    }

    Tensor::TensorCount--;
}

Tensor& Tensor::cpu() {
    if (this->device == CPU) {
        return *this;
    } else {
        Tensor* pTensor_cpu = new Tensor(this->shape, CPU);
        cudaMemcpy(pTensor_cpu->data, this->data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
        return *pTensor_cpu;
    }
}
        
Tensor& Tensor::gpu() {
    if (this->device == GPU) {
        return *this;
    } else {
        Tensor* pTensor_cpu = new Tensor(this->shape, GPU);
        cudaMemcpy(pTensor_cpu->data, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);
        return *pTensor_cpu;
    }
}
/* End Tensor definition */


/* Begin activation definition */
__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

void relu(const Tensor& input, Tensor& output) {
    assert (input.device == GPU);
    assert (output.device == GPU);
    assert (input.shape == output.shape);
    relu_kernel<<<1, input.size>>>(input.data, output.data, input.size);
}

__global__ void sigmoid_kernel(float* input, float* output, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        output[idx] = 1 / (1 + exp(-input[idx]));
    }
}

void sigmoid(const Tensor& input, Tensor& output) {
    assert (input.device == GPU);
    assert (output.device == GPU);
    assert (input.shape == output.shape);

    sigmoid_kernel<<<1, input.size>>>(input.data, output.data, input.size);
}

__global__ void relu_backward_kernel(float* input, float* grad_output, float* grad_input, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
    }
}

void relu_backward(const Tensor& input, Tensor& grad_output, const Tensor& grad_input) {
    assert (input.device == GPU);
    assert (grad_output.device == GPU);
    assert (grad_input.device == GPU);
    assert (input.shape == grad_output.shape);
    assert (input.shape == grad_input.shape);

    relu_backward_kernel<<<1, input.size>>>(input.data, grad_output.data, grad_input.data, input.size);
}

__global__ void sigmoid_backward_kernel(float* input, float* grad_output, float* grad_input, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (1 - 1 / (1 + exp(-input[idx]))) * grad_output[idx] / (1 + exp(-input[idx]));
    }
}

void sigmoid_backward(const Tensor& input, Tensor& grad_output, const Tensor& grad_input) {
    assert (input.device == GPU);
    assert (grad_output.device == GPU);
    assert (grad_input.device == GPU);
    assert (input.shape == grad_output.shape);
    assert (input.shape == grad_input.shape);

    sigmoid_backward_kernel<<<1, input.size>>>(input.data, grad_output.data, grad_input.data, input.size);
}
/* End activation definition */


/* Example */
int main () {
    int size = 10;
    shape_t shape = {size};
    Tensor input(shape, CPU);
    Tensor output(shape, GPU);
    Tensor grad_output(shape, CPU);
    Tensor grad_input(shape, GPU);
    for (int i = 0; i < input.size; i++) {
        input.data[i] = i - 5;
        grad_output.data[i] = 2; // Suppose the gradient with respect to the output equals to 2
    }
    relu(input.gpu(), output);
    relu_backward(input.gpu(), grad_output.gpu(), grad_input);
    output = output.cpu();
    grad_input = grad_input.cpu();

    // Check the results of forward and backward manually
    for (int i = 0; i < size; i++) {
        printf("input=%f, output=%f, grad=%f\n", input.data[i], output.data[i], grad_input.data[i]);
    }

    return 0;
}
