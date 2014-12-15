#include "cmt.h"
#include "utils.h"
#include <iostream>

CMT::CMT::CMT(bool          estimateRotation,
              bool          estimateScale,
              std::string   detectorType,
              std::string   descriptorType,
              int           descriptorLength,
              std::string   matcherType,
              int           threshOutlier,
              float         threshConf,
              float         threshRatio)
{
    m_estimateRotation = estimateRotation;
    m_estimateScale = estimateScale;
    m_detectorType = detectorType;
    m_descriptorType = descriptorType;
    m_descriptorLength = descriptorLength;
    m_matcherType = matcherType;
    m_threshOutlier = threshOutlier;
    m_threshConf = threshConf;
    m_threshRatio = threshRatio;
}

CMT::CMT::~CMT()
{

}

void CMT::CMT::initialise(cv::Mat &initialImageGray,
                          cv::Rect boundingBox)
{
    m_detector = cv::FeatureDetector::create(m_detectorType);
    m_descriptorExtractor = cv::DescriptorExtractor::create(m_descriptorType);
    m_descriptorMatcher = cv::DescriptorMatcher::create(m_matcherType);

    // Detect initial keypoints for the first frame
    std::vector<cv::KeyPoint> keypoints;
    m_detector->detect(initialImageGray, keypoints);

    // Remember the keypoints that are in the target bounding box
    std::vector<cv::KeyPoint> selectedKeypoints;
    std::vector<cv::KeyPoint> backgroundKeypoints;



    // Describe initial keypoints

}

void CMT::CMT::estimate(const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIn,
                        cv::Point2f &center,
                        float &scaleEstimate,
                        float &rotationEstimate,
                        std::vector<std::pair<cv::KeyPoint, int> > &keypointsOut)
{

}

void CMT::CMT::processFrame(cv::Mat imageGray)
{

}
