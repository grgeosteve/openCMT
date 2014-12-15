#ifndef CMT_H
#define CMT_H

#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <string>

namespace CMT
{
    #define CMT_SUCCESS 1
    #define CMT_FAILURE 0

    class CMT
    {

    public:
        CMT(bool        estimateRotation,   // Estimate rotation of target?
            bool        estimateScale,      // Estimate target scale?
            std::string detectorType,
            std::string descriptorType,
            int         descriptorLength,
            std::string matcherType,
            int         threshOutlier,      // Threshold used for clustering votes
            float       threshConf,         // Threshold used in keypoint matching
            float       threshRatio         // Threshold for the ratio of consensus cluster
            );
        ~CMT();
        void initialise(cv::Mat     &initialImageGray,  // Initial grayscale frame
                        cv::Rect    boundingBox,        // Target bounding box
                        int         &result,
                        std::string &errorMessage);
        void processFrame(cv::Mat imageGray);
        void estimate(const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIn,
                      cv::Point2f                                      &center,
                      float                                            &scaleEstimate,
                      float                                            &rotationEstimate,
                      std::vector<std::pair<cv::KeyPoint, int> >       &keypointsOut);

    private:
        std::string m_detectorType;
        std::string m_descriptorType;
        std::string m_matcherType;
        int         m_descriptorLength;
        int         m_threshOutlier;
        float       m_threshConf;
        float       m_threshRatio;
        bool        m_estimateScale;
        bool        m_estimateRotation;

        cv::Ptr<cv::FeatureDetector>     m_detector;
        cv::Ptr<cv::DescriptorExtractor> m_descriptorExtractor;
        cv::Ptr<cv::DescriptorMatcher>   m_descriptorMatcher;

        cv::Mat          m_selectedFeatures;
        std::vector<int> m_selectedClasses;
        cv::Mat          m_featuresDatabase;
        std::vector<int> m_classesDatabase;

        std::vector<std::vector<float> > m_squareFormDists;
        std::vector<std::vector<float> > m_squareFormAngles;

        cv::Point2f m_centerToTopLeft;
        cv::Point2f m_centerToTopRight;
        cv::Point2f m_centerToBottomRight;
        cv::Point2f m_centerToBottomLeft;

        cv::Mat m_previousImageGray;

        std::vector<std::pair<cv::KeyPoint, int> > m_activeKeypoints;
        std::vector<std::pair<cv::KeyPoint, int> > m_trackedKeyPoints;

        int numInitialKeypoints;

        cv::vector<cv::Point2f> m_votes;

        std::vector<std::pair<cv::KeyPoint, int> > outliers;
    };

}
#endif // CMT_H
