#include <vector>
#include "caffe/layers/label_ranking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LabelRankingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
    int N = bottom[0]->num();
    const Dtype * S_data = bottom[0]->gpu_data();
    const Dtype * Y_data = bottom[1]->gpu_data();

    ///D = Y - S
    caffe_copy(N*c_, Y_data, bottom_D_.mutable_gpu_data());
    caffe_gpu_axpy(N*c_, Dtype(-1), S_data, bottom_D_.mutable_gpu_data());
    /// C = (num_neg - num_pos)/num_num * Y + 1/num_neg
    caffe_gpu_scale(N*c_, C_alpha_, Y_data, bottom_C_.mutable_gpu_data());
    caffe_gpu_add_scalar(N*c_, Dtype(1.0/num_neg_), bottom_C_.mutable_gpu_data());

    /// A = (D*Y)1
    caffe_gpu_mul(N*c_, bottom_D_.gpu_data(), Y_data, bottom_tmp1_.mutable_gpu_data());
    caffe_gpu_gemv(CblasNoTrans, N, c_, Dtype(1), bottom_tmp1_.gpu_data(),
            ones_c_.gpu_data(), Dtype(0), bottom_A_.mutable_gpu_data());
    /// B = ((1-Y)*D)1
    caffe_gpu_scale(N*c_, Dtype(-1), Y_data, bottom_Y1_.mutable_gpu_data());
    caffe_gpu_add_scalar(N*c_, Dtype(1), bottom_Y1_.mutable_gpu_data());
    caffe_gpu_mul(N*c_, bottom_Y1_.gpu_data(), bottom_D_.gpu_data(),
            bottom_tmp1_.mutable_gpu_data());
    caffe_gpu_gemv(CblasNoTrans, N, c_, Dtype(1.), bottom_tmp1_.gpu_data(),
            ones_c_.gpu_data(), Dtype(0.), bottom_B_.mutable_gpu_data());
    /// loss = 1/N*sum(D*C*D) + 2/N/mul <A,B>
    caffe_gpu_mul(N*c_, bottom_D_.gpu_data(), bottom_C_.gpu_data(),
            bottom_DC_.mutable_gpu_data());
    caffe_gpu_mul(N*c_, bottom_DC_.gpu_data(), bottom_D_.gpu_data(),
            bottom_tmp1_.mutable_gpu_data());
    Dtype dot = 0;
    caffe_gpu_dot(N*c_, bottom_tmp1_.gpu_data(), ones_Nc_.gpu_data(), &dot);
    Dtype loss = 1.0*dot/N;
    caffe_gpu_dot(N, bottom_A_.gpu_data(), bottom_B_.gpu_data(), &dot);
    loss += Dtype(-2.0/N/num_mul_) * dot;

    top[0]->mutable_cpu_data()[0] = loss;
}    




template <typename Dtype>
void LabelRankingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*> & top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*> & bottom) {
        if(propagate_down[1]) {
            LOG(FATAL)<< this->type()
                    <<" Layer cannot backpropagate to label inputs";
        }
        if(propagate_down[0]) {
            int N = bottom[0]->num();
            Dtype alpha = -2.0/N;
            Dtype beta = 2.0/N/num_mul_;
            /// Y
            caffe_copy(N*c_, bottom[1]->gpu_data(), bottom_Y_.mutable_gpu_data());
            /// -2/N*D*C
            caffe_gpu_scale(N*c_, alpha, bottom_DC_.gpu_data(), bottom[0]->mutable_gpu_diff());
            /// A*Y1 B*Y
            caffe_gpu_vm_mul(N, bottom_A_.gpu_data(), c_, bottom_Y1_.mutable_gpu_data());
            caffe_gpu_vm_mul(N, bottom_B_.gpu_data(), c_, bottom_Y_.mutable_gpu_data());
            /// A*Y1 + B*Y
            caffe_gpu_add(N*c_, bottom_Y1_.gpu_data(), bottom_Y_.gpu_data(),
                    bottom_tmp1_.mutable_gpu_data());
            /// grad = -2/N*D*C +2/N/mul*(A*Y1 + B*Y)
            caffe_gpu_axpy(N*c_, beta, bottom_tmp1_.gpu_data(), bottom[0]->mutable_gpu_diff());
        }        
}

INSTANTIATE_LAYER_GPU_FUNCS(LabelRankingLossLayer);


} //! namespace caffe
