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

void CMT::CMT::initialise(cv::Mat     &initialImageGray,
                          cv::Rect    boundingBox,
                          int         &result,
                          std::string &errorMessage)
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

    inRect(boundingBox, keypoints, selectedKeypoints, backgroundKeypoints);

    if (selectedKeypoints.size() > 0)
    {
        // Describe initial keypoints
        // Describe selected keypoints
        cv::Mat selectedFeatures;
        m_descriptorExtractor->compute(initialImageGray, selectedKeypoints, selectedFeatures);

        // Describe background keypoints
        cv::Mat backgroundFeatures;
        m_descriptorExtractor->compute(initialImageGray, backgroundKeypoints, backgroundFeatures);

        // Insert selected and background features into a features database
        int maxCols = (backgroundFeatures.cols > selectedFeatures.cols) ? (backgroundFeatures.cols) : (selectedFeatures.cols);
        cv::Mat featuresDatabase = cv::Mat((backgroundFeatures.rows+selectedFeatures.rows), maxCols, selectedFeatures.type());
        if (backgroundFeatures.cols > 0)
        {
            backgroundFeatures.copyTo(featuresDatabase(cv::Rect(
                                0, 0, backgroundFeatures.cols,
                                backgroundFeatures.rows)));
        }
        if (selectedFeatures.cols > 0)
        {
            selectedFeatures.copyTo(featuresDatabase(cv::Rect(
                                0, backgroundFeatures.rows,
                                selectedFeatures.cols,
                                selectedFeatures.rows)));
        }
        m_selectedFeatures = selectedFeatures.clone();
        m_featuresDatabase = featuresDatabase.clone();

        // Assign classes to detected keypoints
        // For selected keypoints start from 1 and increase for each keypoint
        // For background keypoints class is 0
        m_classesDatabase = std::vector<int>();
        for (int i = 0; i < backgroundKeypoints.size(); i++)
        {
            m_classesDatabase.push_back(0);
        }
        for (int i = 0; i < selectedKeypoints.size(); i++)
        {
            m_selectedClasses.push_back(i+1);
            m_classesDatabase.push_back(i+1);
        }

        // Get all distances and angles between selected keypoints


        result = CMT_SUCCESS;
    }
    else {
        result = CMT_FAILURE;
        errorMessage = "ERROR: No keypoints found in selection";
        return;
    }

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
