#include <vector>

#include "caffe/layers/cosine_similarity_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename DType>
void CosineSimilarityLayer<DType>::Reshape(const vector<Blob<DType>*>& bottom,
        const vector<Blob<DType>*>& top) {
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  vector<int> top_shape(0); // scalar; 0 axes
  top[0]->Reshape(top_shape);

}

template <typename DType>
void CosineSimilarityLayer<DType>::Forward_cpu(const vector<Blob<DType>* >& bottom, 
	const vector<Blob<DType>* > & top) {
	int num = bottom[0]->num();	
	int dim = bottom[0]->count(1);
	const DType * x1 = bottom[0]->cpu_data();
	const DType * x2 = bottom[1]->cpu_data();

	DType loss = 0, dot, norm1, norm2;

	for(int i = 0; i < num; ++i) {
//        int offset = bottom[0]->offset(i);
        int offset = i*dim;
		dot = caffe_cpu_dot(dim, x1 + offset, x2 + offset);
		norm1 = caffe_cpu_norm2(dim, x1+offset);
		norm2 = caffe_cpu_norm2(dim, x2+offset);
		loss += dot/norm1/norm2;
	}	
	loss /= num;
	top[0]->mutable_cpu_data()[0] = loss;
}







#ifdef CPU_ONLY
STUB_GPU(CosineSimilarityLayer);
#endif

INSTANTIATE_CLASS(CosineSimilarityLayer);
REGISTER_LAYER_CLASS(CosineSimilarity);

} //! namespace caffe
