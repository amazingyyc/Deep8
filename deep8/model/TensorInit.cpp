#include "Device.h"
#include "TensorUtils.h"
#include "TensorInit.h"

namespace Deep8 {

template <typename T>
void TensorInit<T>::constantCPU(Tensor<T> &tensor, T v) {
	auto device = static_cast<CPUDevice*>(tensor.device())->eigenDevice;

	auto size = (int64_t)tensor.size();

	int64_t threadNum = device->numThreads();
	int64_t blockSize = (size + threadNum - 1) / threadNum;

	Eigen::Barrier barrier(static_cast<unsigned int>(threadNum));

	auto blockFunc = [&](T *value, int64_t start, int64_t end) {
		for (int64_t i = start; i < end; ++i) {
			value[i] = v;
		}

		barrier.Notify();
	};

	for (int64_t i = 0; i < threadNum; ++i) {
		int64_t start = i * blockSize;
		int64_t end = std::min<int64_t>(start + blockSize, size);

		device->enqueueNoNotification(blockFunc, tensor.data(), start, end);
	}

	barrier.Wait();
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::constantCPU(Tensor<half> &tensor, half v) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

template <typename T>
 void TensorInit<T>::uniformCPU(Tensor<T> &tensor, T left, T right) {
	auto device = static_cast<CPUDevice*>(tensor.device());

	std::uniform_real_distribution<T> distribution(left, right);

	std::generate(tensor.data(), tensor.data() + tensor.size(), std::bind(distribution, device->randGenerator));
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::uniformCPU(Tensor<half> &tensor, half left, half right) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

template <typename T>
void TensorInit<T>::gaussianCPU(Tensor<T> &tensor, T mean, T stddev) {
	auto device = static_cast<CPUDevice*>(tensor.device());

	std::normal_distribution<T> distribution(mean, stddev);

	std::generate(tensor.data(), tensor.data() + tensor.size(), std::bind(distribution, device->randGenerator));
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::gaussianCPU(Tensor<half> &tensor, half mean, half stddev) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

template <typename T>
void TensorInit<T>::positiveUnitballCPU(Tensor<T> &tensor) {
	auto device = static_cast<CPUDevice*>(tensor.device())->eigenDevice;

	uniformCPU(tensor, 0, 1);

	T sum = 0;

	Eigen::array<size_t, 1> sumDims = { 0 };

	Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> sumTensor(&sum, 1);
	sumTensor.device(*device) = eTVec(tensor).sum(sumDims);

	eTVec(tensor).device(*device) = eTVec(tensor) / sum;
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::positiveUnitballCPU(Tensor<half> &tensor) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

DEEP8_DECLARATION_INSTANCE(TensorInit)

}