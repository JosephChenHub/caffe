#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multi_label_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename TypeParam>
class MultiLabelDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiLabelDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg," << i << std::endl;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg," << 0 << std::endl;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg," << 1
                << std::endl;
    reshapefile.close();
    // Create test input file for images with space in names
    MakeTempFilename(&filename_space_);
    std::ofstream spacefile(filename_space_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_space_;
    spacefile << EXAMPLES_SOURCE_DIR "images/cat.jpg," << 0 << std::endl;
    spacefile << EXAMPLES_SOURCE_DIR "images/cat gray.jpg," << 1 << std::endl;
    spacefile.close();
    /// Create test input file for multi-labels
    MakeTempFilename(&filename_multi_);
    std::ofstream multifile(filename_multi_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_multi_;
    for (int i = 0; i < 5; ++i) {
      multifile << EXAMPLES_SOURCE_DIR "images/cat.jpg," << i+0.5
                <<","<< i+1.5 <<"," <<i+3.5 << std::endl;
    }
    multifile.close();

  }

  virtual ~MultiLabelDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  string filename_space_;
  string filename_multi_; 
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiLabelDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiLabelDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MultiLabelDataParameter* multi_label_data_param = param.mutable_multi_label_data_param();
  multi_label_data_param->set_batch_size(5);
  multi_label_data_param->set_source(this->filename_.c_str());
  multi_label_data_param->set_shuffle(false);
  MultiLabelDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);


  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(MultiLabelDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MultiLabelDataParameter* multi_label_data_param = param.mutable_multi_label_data_param();
  multi_label_data_param->set_batch_size(5);
  multi_label_data_param->set_source(this->filename_.c_str());
  multi_label_data_param->set_new_height(256);
  multi_label_data_param->set_new_width(256);
  multi_label_data_param->set_shuffle(false);
  MultiLabelDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(MultiLabelDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MultiLabelDataParameter* multi_label_data_param = param.mutable_multi_label_data_param();
  multi_label_data_param->set_batch_size(1);
  multi_label_data_param->set_source(this->filename_reshape_.c_str());
  multi_label_data_param->set_shuffle(false);
  MultiLabelDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
}

TYPED_TEST(MultiLabelDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MultiLabelDataParameter* multi_label_data_param = param.mutable_multi_label_data_param();
  multi_label_data_param->set_batch_size(5);
  multi_label_data_param->set_source(this->filename_.c_str());
  multi_label_data_param->set_shuffle(true);
  MultiLabelDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

TYPED_TEST(MultiLabelDataLayerTest, TestSpace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MultiLabelDataParameter* multi_label_data_param = param.mutable_multi_label_data_param();
  multi_label_data_param->set_batch_size(1);
  multi_label_data_param->set_source(this->filename_space_.c_str());
  multi_label_data_param->set_shuffle(false);
  MultiLabelDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->cpu_data()[0], 0);
  // cat gray.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->cpu_data()[0], 1);
}

TYPED_TEST(MultiLabelDataLayerTest, TestMultiRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  MultiLabelDataParameter* multi_label_data_param = param.mutable_multi_label_data_param();
  multi_label_data_param->set_batch_size(5);
  multi_label_data_param->set_source(this->filename_multi_.c_str());
  multi_label_data_param->set_shuffle(false);
  MultiLabelDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  cout<<"top label shape:"<<this->blob_top_label_->shape_string()
      <<"top data shape:" << this->blob_top_data_->shape_string()<<endl;

  cout<<"top label:"<< endl;
  int cnt = this->blob_top_label_->count();
  for(int i = 0; i < cnt; ++i) {
      cout<< this->blob_top_label_->cpu_data()[i] << " ";
  }
  cout<<endl;

  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);

  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 3);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    
    cout<<"top label2:"<< endl;
    int cnt = this->blob_top_label_->count();
    for(int i = 0; i < cnt; ++i) {
          cout<< this->blob_top_label_->cpu_data()[i] << " ";
    }

    for (int i = 0; i < 5; ++i) {
        int offset = this->blob_top_label_->offset(i);
        EXPECT_NEAR(i+0.5, this->blob_top_label_->cpu_data()[offset], 1e-5);
        EXPECT_NEAR(i+1.5, this->blob_top_label_->cpu_data()[offset+1], 1e-5);
        EXPECT_NEAR(i+3.5, this->blob_top_label_->cpu_data()[offset+2], 1e-5);
    }
  }

  // show image
/*
  vector<cv::Mat> ccs;
  cv::Size ss(this->blob_top_data_->width(), this->blob_top_data_->height());
  Dtype * data = this->blob_top_data_->mutable_cpu_data();
  for (int i = 0; i < this->blob_top_data_->channels(); ++i) {
    cv::Mat channel(ss, CV_32FC1, data);
    ccs.push_back(channel);
    data += ss.area();
    }
    cv::Mat res,dst;
//             // merge them
    cv::merge(ccs, res);
//             // optional add mean if needed
     cv::normalize(res, dst, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow("Display window");
    cv::imshow("Display window", dst);
    cv::waitKey(0);
*/

}

}  // namespace caffe
#endif  // USE_OPENCV
