
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/label_ranking_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename TypeParam>
class LabelRankingLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LabelRankingLossLayerTest(): 
      blob_bottom_data_(new Blob<Dtype>(32, 100, 1, 1)),
      blob_bottom_label_(new Blob<Dtype>(32, 100, 1, 1)),
      blob_top_loss_(new Blob<Dtype>()) {
        // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    
    //filler.Fill(this->blob_bottom_label_);
    Dtype *data = blob_bottom_label_->mutable_cpu_data();
    caffe_set(32*100, Dtype(0), data);
    for(int i = 0;i < 32; ++i) {
        data[i*100 + 0] = 1;
        data[i*100 + 10] = 1;
        data[i*100 + 20] = 1;
        data[i*100 + 30] = 1;
        data[i*100 + 40] = 1;
     }
    data[15] =1 ;
    data[25] = 1;
    data[10*100 + 25] = 1;

    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
      }

  virtual ~LabelRankingLossLayerTest() {
      delete blob_bottom_data_;
      delete blob_bottom_label_;
      delete blob_top_loss_;
  }

  void TestForward() {
    LayerParameter layer_param;
    LabelRankingLossParameter * param = layer_param.mutable_label_ranking_loss_param();    
    int N = 32, c = 100, k = 5;
    param->set_top_k(k);
    param->set_num_output(c);

    LabelRankingLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss1 = layer_weight_1.Forward(this->blob_bottom_vec_, 
                this->blob_top_vec_);
//    Dtype loss1 = this->blob_top_vec_[0]->data_at(0,0,0,0);

    const Dtype kLossWeight = 3.8;
    layer_param.add_loss_weight(kLossWeight);
    LabelRankingLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss2 = layer_weight_2.Forward(this->blob_bottom_vec_, 
                this->blob_top_vec_);
//    Dtype loss2 = this->blob_top_vec_[0]->data_at(0,0,0,0);
    
    EXPECT_NEAR(loss1*kLossWeight, loss2, 1e-4);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(LabelRankingLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelRankingLossLayerTest, TestForward) {
    this->TestForward();
}

TYPED_TEST(LabelRankingLossLayerTest, TestGradient) {

    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    LabelRankingLossParameter * param = layer_param.mutable_label_ranking_loss_param();    
    int N = 32, c = 100, k = 5;
    param->set_top_k(k);
    param->set_num_output(c);
    Dtype w = 1.0;
    layer_param.add_loss_weight(w);
    LabelRankingLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_,0);
}


} //! namespace 
