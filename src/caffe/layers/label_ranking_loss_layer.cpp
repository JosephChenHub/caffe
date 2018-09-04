#include <vector>

#include "caffe/layers/label_ranking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template<typename Dtype>
void LabelRankingLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> & bottom,
        const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top); //! add_loss_weight 1
    top_k_ = this->layer_param_.label_ranking_loss_param().top_k();
    c_ = this->layer_param_.label_ranking_loss_param().num_output();

    num_pos_ = top_k_;
    num_neg_ = c_ - num_pos_;
    num_mul_ = num_pos_ * num_neg_;
    C_alpha_ = Dtype((num_neg_ - num_pos_)*1.0/num_mul_);

    ones_c_.Reshape(1,1,c_,1);
    caffe_set(c_, Dtype(1), ones_c_.mutable_cpu_data());

}

template <typename Dtype>
void LabelRankingLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
        << "Inputs must have the same dimension.";
    CHECK_EQ(bottom[0]->count(1), c_) 
        << "num_output must be same with Inputs dimension";
    vector<int> top_shape(0); // scalar; 0 axes
    top[0]->Reshape(top_shape);

    int N = bottom[0]->num();
    bottom[0]->Reshape(N, c_, 1, 1);
    bottom[1]->Reshape(N, c_, 1, 1);

    bottom_A_.Reshape(N, 1, 1, 1);
    bottom_B_.Reshape(N, 1, 1, 1);
    bottom_C_.Reshape(N, c_, 1, 1);
    bottom_D_.Reshape(N, c_, 1, 1);
    bottom_Y1_.Reshape(N, c_, 1, 1);
    bottom_Y_.Reshape(N,c_, 1, 1);
    bottom_tmp1_.Reshape(1, 1,N, c_);
    bottom_DC_.Reshape(1,1, N, c_);
    ones_Nc_.Reshape(1,1, N*c_, 1);
    caffe_set(N*c_, Dtype(1), ones_Nc_.mutable_cpu_data());

#if 1
    const Dtype * label = bottom[1]->cpu_data();

    bottom_C_alpha_.Reshape(N,1, 1,  1);
    bottom_C_beta_.Reshape(N, 1, 1, 1);
    bottom_inv_mul_.Reshape(N, 1, 1,1);
    int offset = 0;
    for(int i = 0; i < N; ++i) {
        offset = bottom[1]->offset(i);
        Dtype pos = 0;
        for(int j = 0; j < c_; ++j) {
            pos += label[offset + j]; 
        }
        Dtype neg = c_ - pos;
        bottom_C_alpha_.mutable_cpu_data()[i] = (neg - pos)/pos/neg;
        bottom_C_beta_.mutable_cpu_data()[i] = 1.0/neg;
        bottom_inv_mul_.mutable_cpu_data()[i] = 1.0/neg/pos;
    }
#endif

}

template <typename Dtype>
void LabelRankingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom, 
	const vector<Blob<Dtype>* > & top) {
    const Dtype * S_data = bottom[0]->cpu_data();
    const Dtype * Y_data = bottom[1]->cpu_data();
    int N = bottom[0]->num();
#if 0
    std::cout<<"Batch:"<<N <<" c:" << c_
        << " norm(S):"<<caffe_cpu_norm2(N*c_, S_data) 
        <<" norm(Y):" << caffe_cpu_norm2(N*c_, Y_data) << std::endl;
    for(int i = 0; i < 5; ++i) {
        std::cout<<S_data[i] << ",";
    }
    std::cout<<std::endl;
#endif 
    /// D = Y-S, shape Nxc
    caffe_copy(N*c_, Y_data, bottom_D_.mutable_cpu_data());
    caffe_axpy(N*c_, Dtype(-1), S_data, bottom_D_.mutable_cpu_data()); //! D=Y-S
    /// C = (num_neg - num_pos)/num_mul* Y + 1/num_neg
    /// C = alpha*Y+beta
#if 0     
    caffe_cpu_scale(N*c_, C_alpha_, Y_data, bottom_C_.mutable_cpu_data());
    caffe_add_scalar(N*c_, Dtype(1.0/num_neg_), bottom_C_.mutable_cpu_data());
#else    
    caffe_copy(N*c_, bottom[1]->cpu_data(), bottom_C_.mutable_cpu_data());
    caffe_cpu_vm_mul(N, bottom_C_alpha_.cpu_data(), c_, bottom_C_.mutable_cpu_data());
    caffe_cpu_vm_add(N, bottom_C_beta_.cpu_data(), c_, bottom_C_.mutable_cpu_data());
#endif 

    /// A = (D\odot Y)1
    caffe_mul(N*c_, bottom_D_.cpu_data(), Y_data, bottom_tmp1_.mutable_cpu_data());

    caffe_cpu_gemv(CblasNoTrans, N, c_, Dtype(1), bottom_tmp1_.cpu_data(),ones_c_.cpu_data(), Dtype(0), 
            bottom_A_.mutable_cpu_data());

    /// B = ((1-Y)\odot D)1
    caffe_cpu_scale(N*c_, Dtype(-1), Y_data, bottom_Y1_.mutable_cpu_data());
    caffe_add_scalar(N*c_, Dtype(1), bottom_Y1_.mutable_cpu_data());
    caffe_mul(N*c_, bottom_Y1_.cpu_data(), bottom_D_.cpu_data(), 
            bottom_tmp1_.mutable_cpu_data());
    caffe_cpu_gemv(CblasNoTrans, N, c_, Dtype(1.), bottom_tmp1_.cpu_data(), ones_c_.cpu_data(), Dtype(0),
            bottom_B_.mutable_cpu_data());
   /// loss = 1/N* tr((D*C)D^T) + 2/N/mul <A,B>
    caffe_mul(N*c_, bottom_D_.cpu_data(), bottom_C_.cpu_data(),
            bottom_DC_.mutable_cpu_data());
    caffe_mul(N*c_, bottom_DC_.cpu_data(), bottom_D_.cpu_data(), 
            bottom_tmp1_.mutable_cpu_data());

    Dtype loss = 1.0/N*caffe_cpu_dot(N*c_, bottom_tmp1_.cpu_data(), ones_Nc_.cpu_data());
#if 0 
#else
    caffe_mul(N, bottom_A_.cpu_data(), bottom_inv_mul_.cpu_data(), bottom_A_.mutable_cpu_data());
#endif    
    loss += Dtype(-2.0/N/num_mul_)*caffe_cpu_dot(N, bottom_A_.cpu_data(), 
            bottom_B_.cpu_data());

    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void LabelRankingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* > & top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  ///grad = -2/N D*C + 2/N/mul (A*Y1 + B*Y)
  if(propagate_down[0]) {
    int N = bottom[0]->num();

    Dtype alpha = Dtype(-2.0/N)*2.0;
    /// Y
    caffe_copy(N*c_, bottom[1]->cpu_data(), bottom_Y_.mutable_cpu_data());
    /// -2/N*D*C
    caffe_cpu_scale(N*c_, alpha, bottom_DC_.cpu_data(), bottom[0]->mutable_cpu_diff());
    /// A*Y1  B*Y
    for(int i = 0; i < N; ++i) {
        caffe_scal(c_, bottom_A_.cpu_data()[i], bottom_Y1_.mutable_cpu_data() + i*c_);
        caffe_scal(c_, bottom_B_.cpu_data()[i], bottom_Y_.mutable_cpu_data() + i*c_);
    }
#if 1
    caffe_cpu_vm_mul(N, bottom_inv_mul_.cpu_data(), c_, bottom_Y1_.mutable_cpu_data() );
    caffe_cpu_vm_mul(N, bottom_inv_mul_.cpu_data(), c_, bottom_Y_.mutable_cpu_data());
    Dtype beta = 2.0/N*2;
#else
    Dtype beta = 2.0/N/num_mul_;
#endif    

    /// A*Y1+B*Y
    caffe_add(N*c_, bottom_Y1_.cpu_data(), bottom_Y_.cpu_data(), 
            bottom_tmp1_.mutable_cpu_data());
    caffe_axpy(N*c_, beta, bottom_tmp1_.cpu_data(), bottom[0]->mutable_cpu_diff());

  }  
    

}






#ifdef CPU_ONLY
STUB_GPU(LabelRankingLossLayer);
#endif

INSTANTIATE_CLASS(LabelRankingLossLayer);
REGISTER_LAYER_CLASS(LabelRankingLoss);

} //! namespace caffe
