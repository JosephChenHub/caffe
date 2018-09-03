#include <vector>

#include "caffe/layers/label_ranking_exp_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template<typename Dtype>
void LabelRankingExpLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> & bottom,
        const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top); //! add_loss_weight 1
    top_k_ = this->layer_param_.label_ranking_loss_param().top_k();
    c_ = this->layer_param_.label_ranking_loss_param().num_output();

    num_pos_ = top_k_;
    num_neg_ = c_ - num_pos_;
    num_mul_ = num_pos_ * num_neg_;

}

template <typename Dtype>
void LabelRankingExpLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

}

template <typename Dtype>
void LabelRankingExpLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom, 
	const vector<Blob<Dtype>* > & top) {
    const Dtype * S_data = bottom[0]->cpu_data();
    const Dtype * Y_data = bottom[1]->cpu_data();
    int N = bottom[0]->num();
    Dtype loss = 0,tmp;
    for(int n = 0; n < N; ++n){
        int s_offset = bottom[0]->offset(n);
        int y_offset = bottom[1]->offset(n);

        for(int i = 0; i < num_pos_; ++i) {
            for(int j = num_pos_; j < c_; ++j) {
                int id_i = *(Y_data + y_offset + i);
                int id_j = *(Y_data + y_offset + j);
                tmp = - S_data[s_offset + id_i] + S_data[s_offset + id_j];            
                loss += exp(tmp);
            }
        }
    }
    
    top[0]->mutable_cpu_data()[0] = loss/num_mul_/N;
}

template <typename Dtype>
void LabelRankingExpLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* > & top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype * diff_data = bottom[0]->mutable_cpu_diff();
    const Dtype * Y_data = bottom[1]->cpu_data();
    const Dtype * S_data = bottom[0]->cpu_data();
    int N = bottom[0]->num();

    for(int n = 0; n < N; ++n){
        int offset = bottom[0]->offset(n);
        int y_offset = bottom[1]->offset(n);

        for(int k = 0; k < num_pos_; ++k) {
            Dtype tmp = 0;
            int id_k = Y_data[y_offset + k];
            for(int j = num_pos_; j < c_; ++j){
                int id_j = Y_data[y_offset + j];
                tmp += exp(-S_data[offset+ id_k] + S_data[offset + id_j] );
            }            
            *(diff_data + offset + id_k) = -1.0/num_mul_/N * tmp;
        }

        for(int k = num_pos_; k < c_; ++k) {
            Dtype tmp = 0;
            int id_k = Y_data[y_offset + k];
            for(int i = 0; i < num_pos_; ++i) {
                int id_i = Y_data[y_offset + i];
                tmp += exp(-S_data[offset + id_i] + S_data[offset + id_k]);
            }
            *(diff_data + offset + id_k) = 1.0/num_mul_/N * tmp;
        }
    }

  }

}






#ifdef CPU_ONLY
STUB_GPU(LabelRankingExpLossLayer);
#endif

INSTANTIATE_CLASS(LabelRankingExpLossLayer);
REGISTER_LAYER_CLASS(LabelRankingExpLoss);

} //! namespace caffe
