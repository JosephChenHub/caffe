#ifndef __COSINE_SIMILARITY_LAYER_HPP
#define __COSINE_SIMILARITY_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {
/**
 * @brief Computes the Cosine similarity  @f$
 *   E = \frac{1}{N}\sum_{i=1}^N  
 * \frac{<x_i, y_i>} { ||x_i||_2||y_i||_2  }@f$
 *
 */

template<typename DType>
class CosineSimilarityLayer: public Layer<DType> {
public:
	explicit CosineSimilarityLayer(const LayerParameter& param)
			: Layer<DType>(param) {  }

	virtual void Reshape(const vector<Blob<DType>* >& bottom, 
			const vector<Blob<DType>* > & top);

	virtual inline const char * type() const { return "CosineSimilarity"; } 

	
protected:
	virtual void Forward_cpu(const vector<Blob<DType>* >& bottom, 
			const vector<Blob<DType>* >& top);
	virtual void Forward_gpu(const vector<Blob<DType>* >& bottom, 
			const vector<Blob<DType>* >& top);
	
	virtual void Backward_cpu(const vector<Blob<DType>* > & top, 
            const vector<bool>& propagate_down, const vector<Blob<DType>* >& bottom) {
        for(int i = 0; i < 2; ++i) {
            if(propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }

    }
	virtual void Backward_gpu(const vector<Blob<DType>* > & top, 
		const vector<bool>& propagate_down, const vector<Blob<DType>* >& bottom);


}; //! class
} //! end of namespace 

#endif //! end of file
