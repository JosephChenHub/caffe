#ifndef __LABEL_RANKING_LOSS_EXP_LAYER_HPP
#define __LABEL_RANKING_LOSS_EXP_LAYER_HPP

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
class LabelRankingExpLossLayer: public LossLayer<Dtype> {
public:
	explicit LabelRankingExpLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {  }
    virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
            const vector<Blob<Dtype>*>& top);

	virtual void Reshape(const vector<Blob<Dtype>* >& bottom, 
			const vector<Blob<Dtype>* > & top);

	virtual inline const char * type() const { return "LabelRankingExpLoss"; } 

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
    
    

    int top_k_;
    int c_;
    int num_pos_, num_neg_, num_mul_;

}; //! class
} //! end of namespace 

#endif //! end of file
