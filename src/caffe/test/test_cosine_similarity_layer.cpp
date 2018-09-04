
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cosine_similarity_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CosineSimilarityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CosineSimilarityLayerTest(): 
      blob_bottom_data_(new Blob<Dtype>(20, 1, 1, 15)),
      blob_bottom_label_(new Blob<Dtype>(20, 1, 1, 15)),
      blob_top_loss_(new Blob<Dtype>()) {
        // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
      }

  virtual ~CosineSimilarityLayerTest() {
      delete blob_bottom_data_;
      delete blob_bottom_label_;
      delete blob_top_loss_;
  }

  void TestForward() {
    LayerParameter layer_param;
    CosineSimilarityLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer_weight_1.Forward(this->blob_bottom_vec_, 
                this->blob_top_vec_);
    Dtype sim1 = this->blob_top_loss_->data_at(0,0,0,0);

    LOG(INFO)<<"Cosine--sim1: "<<sim1;
    int num = blob_bottom_vec_[0]->num();
    int dim = blob_bottom_vec_[0]->count(1);
    const Dtype * x1 = blob_bottom_vec_[0]->cpu_data();
    const Dtype * x2 = blob_bottom_vec_[1]->cpu_data();

    Dtype sim2 = 0;
    for(int i = 0; i < num; ++i) {
        Dtype tmp = 0, norm1 = 0, norm2 = 0;
        for(int j = 0; j < dim; ++j) {
            int offset = blob_bottom_vec_[0]->offset(i);

            tmp += x1[offset + j] * x2[offset + j];
            norm1 += x1[offset + j] * x1[offset+j];
            norm2 += x2[offset + j] * x2[offset+j];
        }
        tmp /= sqrt(norm1*norm2);
        sim2 += tmp;
    }
    sim2 /= num;
    LOG(INFO)<<"Cosine--sim2:" << sim2;

    EXPECT_NEAR(sim1, sim2, 1e-4);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(CosineSimilarityLayerTest, TestDtypesAndDevices);

TYPED_TEST(CosineSimilarityLayerTest, TestForward) {
    this->TestForward();
}

TYPED_TEST(CosineSimilarityLayerTest, TestGradient) {


}


}
