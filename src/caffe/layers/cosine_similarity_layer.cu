#include <vector>
#include "caffe/layers/cosine_similarity_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename DType>
void CosineSimilarityLayer<DType>::Forward_gpu(const vector<Blob<DType>*>& bottom, 
        const vector<Blob<DType>*>& top) {
    int num = bottom[0]->num();
    int dim = bottom[0]->count(1);
    const DType * x1 = bottom[0]->gpu_data();
    const DType * x2 = bottom[1]->gpu_data();

    DType sim = 0, dot , norm1, norm2;
    for(int i = 0; i < num; ++i) {
//      int offset = bottom[0]->offset(i);
        int offset = i*dim;

        caffe_gpu_dot(dim, x1+offset, x2+offset, &dot);
        caffe_gpu_norm2(dim, x1+offset, &norm1);
        caffe_gpu_norm2(dim, x2+offset, &norm2);
        sim += dot/norm1/norm2;
    }
    sim /= num;
    top[0]->mutable_cpu_data()[0] = sim;
}    




template <typename DType>
void CosineSimilarityLayer<DType>::Backward_gpu(const vector<Blob<DType>*> & top,
        const vector<bool>& propagate_down, 
        const vector<Blob<DType>*> & bottom) {
    for(int i = 0; i < 2; ++i) {
        if(propagate_down[i]) {
            NOT_IMPLEMENTED;
        }
    }

}

INSTANTIATE_LAYER_GPU_FUNCS(CosineSimilarityLayer);


} //! namespace caffe
