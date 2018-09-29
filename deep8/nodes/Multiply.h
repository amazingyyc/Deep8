#ifndef DEEP8_MULTIPLY_H
#define DEEP8_MULTIPLY_H

namespace Deep8 {

/**
 * @brief this is a element-wise multiply it will BroadCast the input
 */

#ifdef HAVE_CUDA

template <typename real>
__global__ void MultiplyForwardKernel(const real *x, const int *xshape, const int *xdims,
									  const real *y, const int *yshape, const int *ydims,
											real *z, const int *zshape, const int *zdims, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int zIndex[MAX_TENSOR_DIMS];

	for (int i = start; i < N; i += stride) {
		for (int k = 0, index = i; k < MAX_TENSOR_DIMS; ++k) {
			zIndex[k] = index / zdims[k];
			index %= zdims[k];
		}

		int xIndex = 0;
		int yIndex = 0;

		for (int k = 0; k < MAX_TENSOR_DIMS; ++k) {
			if (xshape[k] == zshape[k]) {
				xIndex += zIndex[k] * xdims[k];
			}

			if (yshape[k] == zshape[k]) {
				yIndex += zIndex[k] * ydims[k];
			}
		}

		z[i] = x[xIndex] * y[yIndex];
	}
}

template <typename real>
__global__ void MultiplyBackwardKernel(real *inGrad,  const int *inShape,  const int *inDims,
								 const real *otherValue, const int *otherShape, const int *otherDims,
								 const real *outGrad, const int *outShape, const int *outDims, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int inIndex[MAX_TENSOR_DIMS];
	int otherIndex[MAX_TENSOR_DIMS];
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
				for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
					if (otherShape[l] == outShape[l]) {
						otherIndex[l] = outIndex[l];
					} else {
						otherIndex[l] = 0;
					}
				}

				int otherI = 0;
				int outI = 0;

				for (int l = 0; l < MAX_TENSOR_DIMS; ++l) {
					otherI += otherIndex[l] * otherDims[l];
					outI += outIndex[l] * outDims[l];
				}

				inGrad[inI] += otherValue[otherI] * outGrad[outI];
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

template <typename T>
class Multiply: public Function<T> {
public:
    explicit Multiply(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs dim must be 2 in Multiply Function");

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
            eTVec(output).device(*device) = eTVec(inputs[0]) * eTVec(inputs[1]);
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
                                                                    * eTVec(inputs[1]).reshape(yReshape).broadcast(yBroad);
        }
    }

    template <int diffCount>
    void backwardCPUImpl(Eigen::ThreadPoolDevice *device,
                         const Tensor<T> *otherValue,
                         const Tensor<T> *outputGradient,
                         Tensor<T> *iGradient) {
        auto curElongateDims    = enlongateShapeToMaxDim(iGradient->shape);
        auto otherElongateDims  = enlongateShapeToMaxDim(otherValue->shape);
        auto outputElongateDims = enlongateShapeToMaxDim(outputGradient->shape);

        Eigen::array<int, diffCount> sumDims;

        for (int i = 0, j = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (curElongateDims[i] != outputElongateDims[i]) {
                sumDims[j++] = i;
            }
        }

        auto otherBroad = otherElongateDims;

        for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (otherBroad[i] < outputElongateDims[i]) {
                otherBroad[i] = outputElongateDims[i];
            } else {
                otherBroad[i] = 1;
            }
        }

        eTVec(iGradient).reshape(curElongateDims).device(*device) +=
                ((eTVec(outputGradient).reshape(outputElongateDims)) * (eTVec(*otherValue).reshape(otherElongateDims).broadcast(otherBroad))).sum(sumDims).reshape(curElongateDims);
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

        auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

        auto curValue   = inputs[(0 == index) ? 0 : 1];
        auto otherValue = inputs[(0 == index) ? 1 : 0];

        auto curShape    = curValue->shape;
        auto otherShape  = otherValue->shape;
        auto outputShape = outputGradient->shape;

        if (outputShape == curShape && outputShape == otherShape) {
            eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(*otherValue);
            return;
        }

        auto curElongateDims    = enlongateShapeToMaxDim(curShape);
        auto outputElongateDims = enlongateShapeToMaxDim(outputShape);

        int diffCount = 0;

        for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (curElongateDims[i] != outputElongateDims[i]) {
                diffCount++;
            }
        }

        if (0 == diffCount) {
            backwardCPUImpl<0>(device, otherValue, outputGradient, iGradient);
        } else if (1 == diffCount) {
            backwardCPUImpl<1>(device, otherValue, outputGradient, iGradient);
        } else if (2 == diffCount) {
            backwardCPUImpl<2>(device, otherValue, outputGradient, iGradient);
        } else if (3 == diffCount) {
            backwardCPUImpl<3>(device, otherValue, outputGradient, iGradient);
        } else if (4 == diffCount) {
            backwardCPUImpl<4>(device, otherValue, outputGradient, iGradient);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    }

#ifdef HAVE_CUDA
	template <typename real>
	void forwardGPUImpl(const real *x, const int *xshape, const int *xdims,
						const real *y, const int *yshape, const int *ydims,
							  real *z, const int *zshape, const int *zdims, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MultiplyForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		MultiplyForwardKernel<real> << <grideSize, blockSize >> > (x, xshape, xdims, y, yshape, ydims, z, zshape, zdims, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *x, const int *xshape, const int *xdims,
							  const half *y, const int *yshape, const int *ydims,
									half *z, const int *zshape, const int *zdims, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MultiplyForwardKernel<half> << <grideSize, blockSize >> > (x, xshape, xdims, y, yshape, ydims, z, zshape, zdims, N);
	}
#endif // HAVE_HALF
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		auto device = static_cast<GPUDevice*>(output->device());

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
		int *xdimsPtr = zshapePtr + MAX_TENSOR_DIMS;
		int *ydimsPtr = xdimsPtr + MAX_TENSOR_DIMS;
		int *zdimsPtr = ydimsPtr + MAX_TENSOR_DIMS;

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
	void backwardGPUImpl(real *inGrad,     const int *inShape,    const int *inDims,
				   const real *otherValue, const int *otherShape, const int *otherDims,
				   const real *outGrad,    const int *outShape,   const int *outDims, 
		           const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MultiplyBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		MultiplyBackwardKernel<real> << <grideSize, blockSize >> > (inGrad, inShape, inDims, otherValue, otherShape, otherDims, outGrad, outShape, outDims, N);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(half *inGrad, const int *inShape, const int *inDims,
						 const half *otherValue, const int *otherShape, const int *otherDims,
						 const half *outGrad, const int *outShape, const int *outDims, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MultiplyBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, inShape, inDims, otherValue, otherShape, otherDims, outGrad, outShape, outDims, N);
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

		auto device = static_cast<GPUDevice*>(iGradient->device());

		int inShape[MAX_TENSOR_DIMS];
		int otherShape[MAX_TENSOR_DIMS];
		int outShape[MAX_TENSOR_DIMS];

		int inDims[MAX_TENSOR_DIMS];
		int otherDims[MAX_TENSOR_DIMS];
		int outDims[MAX_TENSOR_DIMS];

		enlongateShapeToMaxDim(iGradient->shape, inShape);
		enlongateShapeToMaxDim(outputGradient->shape, outShape);

		if (0 == index) {
			enlongateShapeToMaxDim(inputs[1]->shape, otherShape);
		} else {
			enlongateShapeToMaxDim(inputs[0]->shape, otherShape);
		}

		inDims[MAX_TENSOR_DIMS - 1]    = 1;
		otherDims[MAX_TENSOR_DIMS - 1] = 1;
		outDims[MAX_TENSOR_DIMS - 1] = 1;

		for (int i = MAX_TENSOR_DIMS - 2; i >= 0; --i) {
			inDims[i]    = inDims[i + 1] * inShape[i + 1];
			otherDims[i] = otherDims[i + 1] * otherShape[i + 1];
			outDims[i]   = outDims[i + 1] * outShape[i + 1];
		}

		auto cudaPtr = (int*)device->malloc(sizeof(int) * MAX_TENSOR_DIMS * 6);
		
		int *inShapePtr    = cudaPtr;
		int *otherShapePtr = inShapePtr + MAX_TENSOR_DIMS;
		int *outShapePtr   = otherShapePtr + MAX_TENSOR_DIMS;

		int *inDimsPtr    = outShapePtr + MAX_TENSOR_DIMS;
		int *otherDimsPtr = inDimsPtr + MAX_TENSOR_DIMS;
		int *outDimsPtr   = otherDimsPtr + MAX_TENSOR_DIMS;

		device->copyFromCPUToGPU(inShape, inShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(otherShape, otherShapePtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(outShape, outShapePtr, sizeof(int) * MAX_TENSOR_DIMS);

		device->copyFromCPUToGPU(inDims, inDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(otherDims, otherDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);
		device->copyFromCPUToGPU(outDims, outDimsPtr, sizeof(int) * MAX_TENSOR_DIMS);

		if (0 == index) {
			backwardGPUImpl(iGradient->data(), inShapePtr, inDimsPtr,
							inputs[1]->data(), otherShapePtr, otherDimsPtr,
							outputGradient->data(), outShapePtr, outDimsPtr,
							iGradient->size());
		} else {
			backwardGPUImpl(iGradient->data(), inShapePtr, inDimsPtr,
							inputs[0]->data(), otherShapePtr, otherDimsPtr,
							outputGradient->data(), outShapePtr, outDimsPtr,
							iGradient->size());
		}

		device->free(cudaPtr);
#else
	DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}



#endif //DEEP8_WISEMULTIPLY_H
