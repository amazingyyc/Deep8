#ifndef DEEP8_ADD_H
#define DEEP8_ADD_H

namespace Deep8 {

/**
 * Z = X + Y
 */

#ifdef HAVE_CUDA

template <typename real> 
__global__ void AddForwardKernel(const real *x, const int *xdims, const int *xstrides,
								 const real *y, const int *ydims, const int *ystrides,
									   real *z, const int *zdims, const int *zstrides, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int zIndex[MAX_TENSOR_DIMS];

	for (int i = start; i < N; i += stride) {
		for (int k = 0, index = i; k < MAX_TENSOR_DIMS; ++k) {
			zIndex[k] = index / zstrides[k];
			index %= zstrides[k];
		}

		int xIndex = 0;
		int yIndex = 0;

		for (int k = 0; k < MAX_TENSOR_DIMS; ++k) {
			if (xdims[k] == zdims[k]) {
				xIndex += zIndex[k] * xstrides[k];
			}

			if (ydims[k] == zdims[k]) {
				yIndex += zIndex[k] * ystrides[k];
			}
		}

		z[i] = x[xIndex] + y[yIndex];
	}
}

template <typename real>
__global__ void AddBackwardKernel(real *inGrad,  const int *inShape,  const int *inDims, 
							const real *outGrad, const int *outShape, const int *outDims, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int inIndex[MAX_TENSOR_DIMS];
	int outIndex[MAX_TENSOR_DIMS];

	for (int inI = start; inI < N; inI += stride) {
		for (int k = 0, index = inI; k < MAX_TENSOR_DIMS; ++k) {
			inIndex[k]  = index / inDims[k];
			outIndex[k] = inIndex[k];

			index %= inDims[k];
		}

		int j = MAX_TENSOR_DIMS - 1;

		while (j >= 0) {
			if (j == MAX_TENSOR_DIMS - 1) {
				int zI = 0;

				for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
					zI += outIndex[l] * outDims[l];
				}

				inGrad[inI] += outGrad[zI];
			}

			if (inShape[j] == outShape[j]) {
				j--;
			} else {
				outIndex[j]++;

				if (outIndex[j] >= outShape[j]) {
					j--;
				} else {
					for (int l = j + 1; l < MAX_TENSOR_DIMS; ++l) {
						outIndex[l] = inIndex[l];
					}

					j = MAX_TENSOR_DIMS - 1;
				}
			}
		}
	}
}

#endif

/**
 * @brief the Add Function, will will Broadcast the input Tensor
 */
template <typename T>
class Add: public Function<T> {
public:
    explicit Add(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs size must be 2 in Add Function");

        /**
         * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
         */
        auto xShape = static_cast<Variable<T>*>(this->inputs[0])->value.shape;
        auto yShape = static_cast<Variable<T>*>(this->inputs[1])->value.shape;

        this->outputShape = broadcastShape(xShape, yShape);
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

        auto xShape = inputs[0]->shape;
        auto yShape = inputs[1]->shape;

        auto zShape = output->shape;

        if (zShape == xShape && zShape == yShape) {
            eTVec(output).device(*device) = eTVec(inputs[0]) + eTVec(inputs[1]);
        } else {
            auto xReshape = enlongateShapeToMaxDim(xShape);
            auto yReshape = enlongateShapeToMaxDim(yShape);
            auto zReshape = enlongateShapeToMaxDim(zShape);

            auto xBroad = xReshape;
            auto yBroad = yReshape;

            for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
                if (xBroad[i] < zReshape[i]) {
                    xBroad[i] = zReshape[i];
                } else {
                    xBroad[i] = 1;
                }

                if (yBroad[i] < zReshape[i]) {
                    yBroad[i] = zReshape[i];
                } else {
                    yBroad[i] = 1;
                }
            }

            eTVec(output).reshape(zReshape).device(*device) = eTVec(inputs[0]).reshape(xReshape).broadcast(xBroad)
                    + eTVec(inputs[1]).reshape(yReshape).broadcast(yBroad);
        }
    }



    template <int diffCount>
    void backwardCPUImpl(Eigen::ThreadPoolDevice *device, const Tensor<T> *outputGradient, Tensor<T> *iGradient) {
        auto outputGradShape = enlongateShapeToMaxDim(outputGradient->shape);
        auto iGradShape      = enlongateShapeToMaxDim(iGradient->shape);

        Eigen::array<int, diffCount> sumDims;

        for (int i = 0, j = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (outputGradShape[i] != iGradShape[i]) {
                sumDims[j++] = i;
            }
        }

        eTVec(iGradient).reshape(iGradShape).device(*device) += eTVec(outputGradient).reshape(outputGradShape).sum(sumDims).reshape(iGradShape);
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

        auto gradShape   = iGradient->shape;
        auto outputShape = outputGradient->shape;

        if (gradShape == outputShape) {
            eTVec(iGradient).device(*device) += eTVec(outputGradient);
            return;
        }

        auto outputGradEnlongateShape = enlongateShapeToMaxDim(outputGradient->shape);
        auto iGradEnlongateShape      = enlongateShapeToMaxDim(iGradient->shape);

        int diffCount = 0;

        for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (outputGradEnlongateShape[i] != iGradEnlongateShape[i]) {
                diffCount++;
            }
        }

        if (1 == diffCount) {
            backwardCPUImpl<1>(device, outputGradient, iGradient);
        } else if (2 == diffCount) {
            backwardCPUImpl<2>(device, outputGradient, iGradient);
        } else if (3 == diffCount) {
            backwardCPUImpl<3>(device, outputGradient, iGradient);
        } else if (4 == diffCount) {
            backwardCPUImpl<4>(device, outputGradient, iGradient);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    }
	
#ifdef HAVE_CUDA

	template <typename real>
	void forwardGPUImpl(const real *x, const int *xdims, const int *xstrides,
						const real *y, const int *ydims, const int *ystrides,
							  real *z, const int *zdims, const int *zstrides, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AddForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AddForwardKernel<real> << <grideSize, blockSize >> > (x, xdims, xstrides, y, ydims, ystrides, z, zdims, zstrides, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *x, const int *xdims, const int *xstrides,
						      const half *y, const int *ydims, const int *ystrides,
									half *z, const int *zdims, const int *zstrides, const int N) {
		int blockSize = 1024;
		int grideSize;

		grideSize = (N + blockSize - 1) / blockSize;

		AddForwardKernel<half> << <grideSize, blockSize >> > (x, xdims, xstrides, y, ydims, ystrides, z, zdims, zstrides, N);
	}

#endif // HAVE_HALF

#endif

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		auto device = static_cast<GPUDevice*>(output->device);
		
		auto x = inputs[0];
		auto y = inputs[1];
		auto z = output;

		int xshape[MAX_TENSOR_DIMS];
		int yshape[MAX_TENSOR_DIMS];
		int zshape[MAX_TENSOR_DIMS];

		int xdims[MAX_TENSOR_DIMS];
		int ydims[MAX_TENSOR_DIMS];
		int zdims[MAX_TENSOR_DIMS];

		enlongateShapeToMaxDim(x->shape, xshape);
		enlongateShapeToMaxDim(y->shape, yshape);
		enlongateShapeToMaxDim(z->shape, zshape);

		xdims[MAX_TENSOR_DIMS - 1] = 1;
		ydims[MAX_TENSOR_DIMS - 1] = 1;
		zdims[MAX_TENSOR_DIMS - 1] = 1;

		for (int i = MAX_TENSOR_DIMS - 2; i >= 0; --i) {
			xdims[i] = xdims[i + 1] * xshape[i + 1];
			ydims[i] = ydims[i + 1] * yshape[i + 1];
			zdims[i] = zdims[i + 1] * zshape[i + 1];
		}

		auto cudaPtr = (int*)device->malloc(sizeof(int) * MAX_TENSOR_DIMS * 6);

		int *xshapePtr = cudaPtr;
		int *yshapePtr = xshapePtr + MAX_TENSOR_DIMS;
		int *zshapePtr = yshapePtr + MAX_TENSOR_DIMS;
		int *xdimsPtr  = zshapePtr + MAX_TENSOR_DIMS;
		int *ydimsPtr  = xdimsPtr  + MAX_TENSOR_DIMS;
		int *zdimsPtr  = ydimsPtr  + MAX_TENSOR_DIMS;

		device->copyFromCPUToGPU(xshape, xshapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(yshape, yshapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(zshape, zshapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(xdims, xdimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(ydims, ydimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(zdims, zdimsPtr, sizeof(int) * MAX_TENSOR_DIMS);

		forwardGPUImpl(x->data(), xshapePtr, xdimsPtr, y->data(), yshapePtr, ydimsPtr, z->data(), zshapePtr, zdimsPtr, static_cast<int>(z->shape.size()));

		device->free(cudaPtr);

#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }

#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(real *inGrad, const int *inShape, const int *inDims, const real *outGrad, const int *outShape, const int *outDims, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AddBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AddBackwardKernel<real> << <grideSize, blockSize >> > (inGrad, inShape, inDims, outGrad, outShape, outDims, N);
	}

#ifdef HAVE_HALF
	template <>
	void backwardGPUImpl<half>(half *inGrad, const int *inShape, const int *inDims, const half *outGrad, const int *outShape, const int *outDims, const int N) {
		int blockSize = 1024;
		int grideSize;

		grideSize = (N + blockSize - 1) / blockSize;

		AddBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, inShape, inDims, outGrad, outShape, outDims, N);
	}
#endif // HAVE_HALF


#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(iGradient->device);

		int inShape[MAX_TENSOR_DIMS];
		int outShape[MAX_TENSOR_DIMS];

		int inDims[MAX_TENSOR_DIMS];
		int outDims[MAX_TENSOR_DIMS];

		enlongateShapeToMaxDim(iGradient->shape, inShape);
		enlongateShapeToMaxDim(outputGradient->shape, outShape);

		inDims[MAX_TENSOR_DIMS - 1] = 1;
		outDims[MAX_TENSOR_DIMS - 1] = 1;

		for (int i = MAX_TENSOR_DIMS - 2; i >= 0; --i) {
			inDims[i]  = inDims[i + 1] * inShape[i + 1];
			outDims[i] = outDims[i + 1] * outShape[i + 1];
		}


		auto cudaPtr = (int*)device->malloc(sizeof(int) * MAX_TENSOR_DIMS * 4);
			
		int *inShapePtr  = cudaPtr;
		int *outShapePtr = inShapePtr  + MAX_TENSOR_DIMS;
		int *inDimsPtr   = outShapePtr + MAX_TENSOR_DIMS;
		int *outDimsPtr  = inDimsPtr   + MAX_TENSOR_DIMS;

		device->copyFromCPUToGPU(inShape,   inShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(outShape, outShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(inDims,   inDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(outDims, outDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);

		backwardGPUImpl(iGradient->data(), inShapePtr, inDimsPtr, 
						outputGradient->data(), outShapePtr, outDimsPtr, static_cast<int>(iGradient->size()));

		device->free(cudaPtr);
#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_ADD_H
