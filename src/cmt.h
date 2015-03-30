#ifndef CMT_H
#define CMT_H

#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <string>

#define CMT_SUCCESS 1
#define CMT_FAILURE 0

class CMT
{
public:
    CMT(bool        estimateRotation    = true,     // Estimate rotation of target?
        bool        estimateScale       = true,     // Estimate target scale?
        std::string detectorType        = "ORB",
        std::string descriptorType      = "ORB",
        int         descriptorLength    = 512,
        std::string matcherType         = "BruteForce-Hamming",
        int         threshOutlier       = 20,       // Threshold used for clustering votes
        float       threshConf          = 0.75,     // Threshold used in keypoint matching
        float       threshRatio         = 0.8       // Threshold for the ratio of consensus clustering
        );

    void initialise(cv::Mat     &initialImageGray,  // Initial grayscale frame
                    cv::Rect    boundingBox,        // Target bounding box
                    int         &result,
                    std::string &errorMesage);

    void processFrame(cv::Mat &imageGray);

    void estimate(const std::vector<std::pair<cv::KeyPoint, int> >  &keypointsIn,
                  cv::Point2f                                       &center,
                  float                                             &scaleEstimate,
                  float                                             &rotationEstimate,
                  std::vector<std::pair<cv::KeyPoint, int> >        &keypointsOut);

    cv::Point2f getCenter();

    float   getScaleEstimate();

    float   getRotationEstimate();

    int     getNumberOfActiveKeypoints();

    void    showAllInfo(cv::Mat &frame);

    // Variables
    std::string m_detectorType;
    std::string m_descriptorType;
    std::string m_matcherType;
    int         m_descriptorLength;
    int         m_threshOutlier;
    float       m_threshConf;
    float       m_threshRatio;
    bool        m_estimateScale;
    bool        m_estimateRotation;

    cv::Ptr<cv::FeatureDetector>        m_detector;
    cv::Ptr<cv::DescriptorExtractor>    m_descriptorExtractor;
    cv::Ptr<cv::DescriptorMatcher>      m_descriptorMatcher;

    cv::Mat             m_selectedFeatures;
    std::vector<int>    m_selectedClasses;
    cv::Mat             m_featuresDatabase;
    std::vector<int>    m_classesDatabase;

    std::vector<std::vector<float> > m_squareFormDists;
    std::vector<std::vector<float> > m_squareFormAngles;

    cv::Point2f m_centerToTopLeft;
    cv::Point2f m_centerToTopRight;
    cv::Point2f m_centerToBottomRight;
    cv::Point2f m_centerToBottomLeft;

    cv::Mat m_previousImageGray;

    std::vector<std::pair<cv::KeyPoint, int> > m_activeKeypoints;
    std::vector<std::pair<cv::KeyPoint, int> > m_trackedKeypoints;

    std::vector<cv::Point2f> m_springs;

    int m_numInitialKeypoints;

    std::vector<cv::Point2f> m_votes;

    cv::Point2f m_center;
    float       m_scaleEstimate;
    float       m_rotationEstimate;

    bool m_hasResult;

    cv::Point2f m_resultBBtopLeft;
    cv::Point2f m_resultBBtopRight;
    cv::Point2f m_resultBBbottomLeft;
    cv::Point2f m_resultBBbottomRight;

    std::vector<std::pair<cv::KeyPoint, int> > m_outliers;
};
#endif // CMT_H
