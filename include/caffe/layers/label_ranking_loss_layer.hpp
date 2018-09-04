#ifndef __LABEL_RANKING_LOSS_LAYER_HPP
#define __LABEL_RANKING_LOSS_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {
/**
 * @brief Label ranking loss  @f$
 *   \ell(k) = \frac{1}{n_{+}n_{-}}\sum_{i\in\mathcal{N}_{+}}
 *   \sum_{j\in\mathcal{N}_{-}} l(x_k, s_i, s_j)
 * @f$
 *
 */

template<typename Dtype>
class LabelRankingLossLayer: public LossLayer<Dtype> {
public:
	explicit LabelRankingLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {  }
    virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
            const vector<Blob<Dtype>*>& top);

	virtual void Reshape(const vector<Blob<Dtype>* >& bottom, 
			const vector<Blob<Dtype>* > & top);

	virtual inline const char * type() const { return "LabelRankingLoss"; } 

    virtual inline int ExactNumBottomBlobs() const { return 2;}
    virtual inline int ExactNumTopBlobs() const { return 1; }
	
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, 
			const vector<Blob<Dtype>* >& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom, 
			const vector<Blob<Dtype>* >& top);
	
	virtual void Backward_cpu(const vector<Blob<Dtype>* > & top, 
            const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);

	virtual void Backward_gpu(const vector<Blob<Dtype>* > & top, 
		const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    
    
    Blob<Dtype> bottom_A_;
    Blob<Dtype> bottom_B_;
    Blob<Dtype> bottom_C_;
    Blob<Dtype> bottom_D_;
    Blob<Dtype> bottom_DC_;
    Blob<Dtype> bottom_Y1_;
    Blob<Dtype> bottom_Y_;
    Blob<Dtype> ones_c_;
    Blob<Dtype> ones_Nc_;
    
    Blob<Dtype> bottom_C_alpha_;
    Blob<Dtype> bottom_C_beta_;
    Blob<Dtype> bottom_inv_mul_;

    Blob<Dtype> bottom_tmp1_;

    int top_k_;
    int c_;
    int num_pos_, num_neg_, num_mul_;
    Dtype C_alpha_;

}; //! class
} //! end of namespace 

#endif //! end of file
