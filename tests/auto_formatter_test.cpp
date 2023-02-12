#include "image_preprocessing_cpp/ImagePreprocessing.hpp"

ImagePreprocessing::ImagePreprocessing(double init_clahe_cliplimit,
                                       double init_clahe_tilegridsize) {
  clahe_ = cv::createCLAHE();
  clahe_->setClipLimit(init_clahe_cliplimit);
  clahe_->setTilesGridSize(
      cv::Size(init_clahe_tilegridsize, init_clahe_tilegridsize));
}

cv::Mat ImagePreprocessing::doClahe(cv::Mat cv_image_, int clahe_plane) {
  cv::cvtColor(cv_image_, lab_image, CV_BGR2Lab);
  std::vector<cv::Mat> lab_planes(3);
  cv::split(lab_image, lab_planes);

  clahe_->apply(lab_planes[clahe_plane], dst);
  dst.copyTo(lab_planes[clahe_plane]);
  cv::merge(lab_planes, lab_image);
  cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  return image_clahe;
}