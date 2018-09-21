#ifndef DEEP8_TENSORINIT_H
#define DEEP8_TENSORINIT_H

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void TensorInitConstantKernel(real *value, real scalar, int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        value[i] = scalar;
    }
}

template <typename real>
__global__ void TensorInitPositiveUnitballKernel(real *value, real sum, int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        value[i] /= sum;
    }
}

#ifdef HAVE_HALF

__global__ void TensorInitConvertFloatToHalf(const float *from, half* to, int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        to[i] = (half) from[i];
    }
}

#endif
#endif

/**
 * init the tensor
 */
class TensorInit {
private:
    template <typename T>
    static void constantCPU(Tensor<T> &tensor, T v) {
        auto device = static_cast<CPUDevice*>(tensor.device)->eigenDevice;

        auto size = (int64_t) tensor.size();

        int64_t threadNum = device->numThreads();
        int64_t blockSize = (size + threadNum - 1) / threadNum;

        Eigen::Barrier barrier(static_cast<unsigned int>(threadNum));

        auto blockFunc = [&] (T *value, int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                value[i] = v;
            }

            barrier.Notify();
        };

        for (int64_t i = 0; i < threadNum; ++i) {
            int64_t start = i * blockSize;
            int64_t end   = std::min<int64_t>(start + blockSize, size);

            device->enqueueNoNotification(blockFunc, tensor.data(), start, end);
        }

        barrier.Wait();
    }


#ifdef HAVE_HALF
	template <>
	static void constantCPU<half>(Tensor<half> &tensor, half v) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF


#ifdef HAVE_CUDA
    template <typename T>
    static void constantGPU(Tensor<T> &tensor, T v) {
        int N = (int)tensor.size();

		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TensorInitConstantKernel<T> << <grideSize, blockSize >> > (tensor.data(), v, N);
    }
#endif

    template <typename T>
    static void uniformCPU(Tensor<T> &tensor, T left, T right) {
        auto device = static_cast<CPUDevice*>(tensor.device);

        std::uniform_real_distribution<T> distribution(left, right);

        std::generate(tensor.data(), tensor.data() + tensor.size(), std::bind(distribution, device->randGenerator));
    }

#ifdef HAVE_HALF
	template <>
	static void uniformCPU<half>(Tensor<half> &tensor, half left, half right) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

#ifdef HAVE_CUDA
    static void uniformGPU(Tensor<float> &tensor) {
        auto device = static_cast<GPUDevice*>(tensor.device);
        CURAND_CHECK(curandGenerateUniform(device->curandGenerator, tensor.data(), (size_t)tensor.size()));
    }

    static void uniformGPU(Tensor<double> &tensor) {
        auto device = static_cast<GPUDevice*>(tensor.device);
        CURAND_CHECK(curandGenerateUniformDouble(device->curandGenerator, tensor.data(), (size_t)tensor.size()));
    }

#ifdef HAVE_HALF
    static void uniformGPU(Tensor<half> &tensor) {
        auto device = static_cast<GPUDevice*>(tensor.device);

        auto size = (size_t) tensor.size();
        auto ptr  = (float*) device->malloc(sizeof(float) * size);

        Tensor<float> tempTensor(ptr, tensor.shape, device);

        uniformGPU(tempTensor);

        int N = (int) size;
        int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TensorInitConvertFloatToHalf<<<grideSize, blockSize >>>(ptr, tensor.data(), N);

        device->free(ptr);
    }   
#endif
#endif

    template <typename T>
    static void gaussianCPU(Tensor<T> &tensor, T mean, T stddev) {
        auto device = static_cast<CPUDevice*>(tensor.device);

        std::normal_distribution<T> distribution(mean, stddev);

        std::generate(tensor.data(), tensor.data() + tensor.size(), std::bind(distribution, device->randGenerator));
    }

#ifdef HAVE_HALF
	template <>
	static void gaussianCPU<half>(Tensor<half> &tensor, half mean, half stddev) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

#ifdef HAVE_CUDA
    static void gaussianGPU(Tensor<float> &tensor, float mean, float stddev) {
        auto device = static_cast<GPUDevice*>(tensor.device);

        CURAND_CHECK(curandGenerateNormal(device->curandGenerator, tensor.data(), (size_t)tensor.size(), mean, stddev));
    }

    static void gaussianGPU(Tensor<double> &tensor, double mean, double stddev) {
        auto device = static_cast<GPUDevice*>(tensor.device);

        CURAND_CHECK(curandGenerateNormalDouble(device->curandGenerator, tensor.data(), (size_t)tensor.size(), mean, stddev));
    }

#ifdef HAVE_HALF
    static void gaussianGPU(Tensor<half> &tensor, half mean, half stddev) {
        auto device = static_cast<GPUDevice*>(tensor.device);

        auto size = (size_t) tensor.size();
        auto ptr  = (float*) device->malloc(sizeof(float) * size);

        Tensor<float> tempTensor(ptr, tensor.shape, device);

		gaussianGPU(tempTensor, __half2float(mean), __half2float(stddev));

        int N = (int) size;
        int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TensorInitConvertFloatToHalf<<<grideSize, blockSize>>>(ptr, tensor.data(), N);

        device->free(ptr);
    }   
#endif
#endif

    template <typename T>
    static void positiveUnitballCPU(Tensor<T> &tensor) {
        auto device = static_cast<CPUDevice*>(tensor.device);

        uniformCPU(tensor, 0, 1);

        T sum = 0;

        Eigen::array<size_t, 1> sumDims = {0};

        Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> sumTensor(&sum, 1);
        sumTensor.device(*device) = eTVec(tensor).sum(sumDims);

        eTVec(tensor).device(*device) = eTVec(tensor) / sum;
    }

#ifdef HAVE_HALF
	template <>
	static void positiveUnitballCPU<half>(Tensor<half> &tensor) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

#ifdef HAVE_CUDA
    static void positiveUnitballGPU(Tensor<float> &tensor) {
        auto device = static_cast<GPUDevice*>(tensor.device);
        int N = (int)tensor.size();

        uniformGPU(tensor);

        float sum = 0;

        CUBLAS_CHECK(cublasSasum(device->cublasHandle, N, tensor.data(), 1, &sum));

        if (0 != sum) {
            int minGrideSize;
            int blockSize;
            int grideSize;

            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TensorInitPositiveUnitballKernel<float>, 0, N));

            grideSize = (N + blockSize - 1) / blockSize;

            TensorInitPositiveUnitballKernel<float> << <grideSize, blockSize >> > (tensor.data(), sum, N);
        }
    }

    static void positiveUnitballGPU(Tensor<double> &tensor) {
        auto device = static_cast<GPUDevice*>(tensor.device);
        int N = (int)tensor.size();

        uniformGPU(tensor);

        double sum = 0;

        CUBLAS_CHECK(cublasDasum(device->cublasHandle, N, tensor.data(), 1, &sum));

        if (0 != sum) {
            int minGrideSize;
            int blockSize;
            int grideSize;

            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TensorInitPositiveUnitballKernel<double>, 0, N));

            grideSize = (N + blockSize - 1) / blockSize;

            TensorInitPositiveUnitballKernel<double> << <grideSize, blockSize >> > (tensor.data(), sum, N);
        }
    }

#ifdef HAVE_HALF
    static void positiveUnitballGPU(Tensor<half> &tensor) {
        auto device = static_cast<GPUDevice*>(tensor.device);

        auto size = (size_t) tensor.size();
        auto ptr  = (float*) device->malloc(sizeof(float) * size);

        Tensor<float> tempTensor(ptr, tensor.shape, device);

        positiveUnitballGPU(tempTensor);

        int N = (int) size;
        int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TensorInitConvertFloatToHalf<<<grideSize, blockSize>>>(ptr, tensor.data(), N);

        device->free(ptr);
    }   
#endif
#endif

public:
    /**set tensor to constant*/
    template <typename T>
    static void constant(Tensor<T> &tensor, T v) {
        if (DeviceType::CPU == tensor.device->type) {
            constantCPU(tensor, v);
        } else {
#ifdef HAVE_CUDA
            constantGPU(tensor, v);
#else
            DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
        }
    }

    template <typename T>
    static void uniform(Tensor<T> &tensor, T left = 0.0, T right = 1.0) {
        if (DeviceType::CPU == tensor.device->type) {
            uniformCPU(tensor, left, right);
        } else {
#ifdef HAVE_CUDA
            uniformGPU(tensor);
#else
            DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
        }
    }

    template <typename T>
    static void gaussian(Tensor<T> &tensor, T mean = 0.0, T stddev = 1.0) {
        if (DeviceType::CPU == tensor.device->type) {
            // TensorInit::gaussianCPU(tensor, mean, stddev);
        } else {
#ifdef HAVE_CUDA
            gaussianGPU(tensor, mean, stddev);
#else
            DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
        }
    }

    template <typename T>
    static void positiveUnitball(Tensor<T> &tensor) {
        if (DeviceType::CPU == tensor.device->type) {
            positiveUnitballCPU(tensor);
        } else {
#ifdef HAVE_CUDA
            positiveUnitballGPU(tensor);
#else
            DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
        }
    }
};

}

#endif //DEEP8_TENSORINIT_H
