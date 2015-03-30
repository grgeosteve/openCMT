#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include "cmt.h"
#include "utils.h"
#include "cluster.h"

CMT::CMT(bool           estimateRotation,
         bool           estimateScale,
         std::string    detectorType,
         std::string    descriptorType,
         int            descriptorLength,
         std::string    matcherType,
         int            threshOutlier,
         float          threshConf,
         float          threshRatio)
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

void CMT::initialise(cv::Mat        &initialImageGray,
                     cv::Rect       boundingBox,
                     int            &result,
                     std::string    &errorMesage)
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

    inRect(boundingBox,
           keypoints,
           selectedKeypoints,
           backgroundKeypoints);

    if (selectedKeypoints.size() > 0)
    {
        // Describe initial keypoints
        // Describe selected keypoints
        cv::Mat selectedFeatures;
        m_descriptorExtractor->compute(initialImageGray,
                                       selectedKeypoints,
                                       selectedFeatures);

        // Describe background keypoints
        cv::Mat backgroundFeatures;
        m_descriptorExtractor->compute(initialImageGray,
                                       backgroundKeypoints,
                                       backgroundFeatures);

        // Insert selected and background features into a features database
        int maxCols = (backgroundFeatures.cols > selectedFeatures.cols) ?
                    (backgroundFeatures.cols) :
                    (selectedFeatures.cols);
        cv::Mat featuresDatabase = cv::Mat(
                    (backgroundFeatures.rows + selectedFeatures.rows),
                    maxCols,
                    selectedFeatures.type());
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
        m_selectedFeatures = selectedFeatures;
        m_featuresDatabase = featuresDatabase;

        // Asign each keypoint a class starting from 1, background is 0
        m_selectedClasses = std::vector<int>();
        for (int i = 1; i <= selectedKeypoints.size(); i++)
        {
            m_selectedClasses.push_back(i);
        }

        std::vector<int> backgroundClasses;
        for (int i = 0; i < backgroundKeypoints.size(); i++)
        {
            backgroundClasses.push_back(0);
        }

        // Assign classes to detected keypoints
        // For selected keypoints start from 1 and increase for each keypoint
        // For background keypoints class is 0
        m_classesDatabase = std::vector<int>();
        for (int i = 0; i < backgroundKeypoints.size(); i++)
        {
            m_classesDatabase.push_back(backgroundClasses[i]);
        }
        for (int i = 0; i < selectedKeypoints.size(); i++)
        {
            m_classesDatabase.push_back(m_selectedClasses[i]);
        }

        // Get all distances and angles between selected keypoints
        std::vector<std::vector<float> > squareFormDists;
        computeSquareformDist(selectedKeypoints, squareFormDists);
        m_squareFormDists = squareFormDists;

        std::vector<std::vector<float> > squareFromAngles;
        computeSquareformAngle(selectedKeypoints, squareFromAngles);
        m_squareFormAngles = squareFromAngles;

        // Compute the center of the selected keypoints
        cv::Point2f center;
        for (int i = 0; i < selectedKeypoints.size(); i++)
        {
            center += selectedKeypoints[i].pt;
        }
        center *= 1.0 / selectedKeypoints.size();

        // Compute the relative position of the bounding box corners to
        // the center of the selected keypoints
        m_centerToTopLeft = cv::Point2f(boundingBox.tl().x, boundingBox.tl().y) - center;
        m_centerToTopRight = cv::Point2f(boundingBox.br().x, boundingBox.tl().y) - center;
        m_centerToBottomRight = cv::Point2f(boundingBox.br().x, boundingBox.br().y) - center;
        m_centerToBottomLeft = cv::Point2f(boundingBox.tl().x, boundingBox.br().y) - center;

        // Make selectedKeypoints (keypoints that lie in the initial bounding box)
        // active and add class information
        m_activeKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();
        for (int i = 0; i < selectedKeypoints.size(); i++)
        {
            m_activeKeypoints.push_back(std::make_pair(selectedKeypoints[i], i + 1));
        }

        // Compute the relative positions of the keypoints to the center
        m_springs = std::vector<cv::Point2f>();
        for (int i = 0; i < selectedKeypoints.size(); i++)
        {
            m_springs.push_back(selectedKeypoints[i].pt - center);
        }

        // Store the initial image as the previous image for tracking
        m_previousImageGray = initialImageGray.clone();

        // Remember the number of initial keypoints
        m_numInitialKeypoints = selectedKeypoints.size();

        result = CMT_SUCCESS;
    }
    else
    {
        result = CMT_FAILURE;
        errorMesage = "ERROR: No keypoints found in selection";
        return;
    }
}

void CMT::processFrame(cv::Mat &imageGray)
{
    // Detect keypoints in the image
    std::vector<cv::KeyPoint> keypoints;
    m_detector->detect(imageGray, keypoints);

    // Describe the detected keypoints
    cv::Mat keypointsFeatures;
    m_descriptorExtractor->compute(imageGray, keypoints, keypointsFeatures);

    // Track the keypoints from the previous frame to the current frame
    std::vector<std::pair<cv::KeyPoint, int> > trackedKeypoints;
    trackLK(imageGray, m_previousImageGray, m_activeKeypoints,
            trackedKeypoints);

    // Create a list of active keypoints
    std::vector<std::pair<cv::KeyPoint, int> > activeKeypoints;

    // Match the candidate keypoints with the initial keypoints
    // Get the best two matches for each feature
    std::vector<std::vector<cv::DMatch> > matchesAll, selectedMatchesAll;
    m_descriptorMatcher->knnMatch(keypointsFeatures, m_featuresDatabase, matchesAll, 2);

    // For each keypoint and its descriptor
    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::KeyPoint keypoint = keypoints[i];

        // First: Match over whole image
        // Compute distances to all descriptors
        std::vector<cv::DMatch> matches = matchesAll[i];

        // Convert distnces to confidences, do not weight
        std::vector<float> combined;
        for (int j = 0; j < matches.size(); j++)
        {
            combined.push_back(1 - matches[j].distance / m_descriptorLength);
        }

        std::vector<int> classes = m_classesDatabase;

        // Get best and second best index
        int bestInd = matches[0].trainIdx;
        int secondBestInd = matches[1].trainIdx;

        // Compute distance ratio according to Lowe
        float ratio = (1 - combined[0]) / (1 - combined[1]);

        // Extract class of best match
        int keypoint_class = classes[bestInd];

        // If distance ratio is ok and absolute distance is ok and keypoint
        // class is not background
        if (ratio < m_threshRatio && combined[0] > m_threshConf && keypoint_class != 0)
        {
            activeKeypoints.push_back(std::make_pair(keypoint, keypoint_class));
        }
    }

    if (trackedKeypoints.size() > 0)
    {
        // Extract the keypoint classes
        std::vector<int> trackedClasses(trackedKeypoints.size());
        for (int i = 0; i < trackedKeypoints.size(); i++)
        {
            trackedClasses[i] = trackedKeypoints[i].second;
        }
        // If there already are some matched keypoints
        if (activeKeypoints.size() > 0)
        {
            // Add all tracked keypoints that have not been matched
            std::vector<int> associatedClasses(activeKeypoints.size());
            for (int i = 0; i < activeKeypoints.size(); i++)
            {
                associatedClasses[i] = activeKeypoints[i].second;
            }

            std::vector<bool> notmissing = inVector(trackedClasses,
                                                    associatedClasses);

            for (int i = 0; i < trackedKeypoints.size(); i++)
            {
                if (!notmissing[i])
                {
                    activeKeypoints.push_back(trackedKeypoints[i]);
                }
            }
        }
        else
        {
            activeKeypoints = trackedKeypoints;
        }
    }

    cv::Point2f center;
    float scaleEstimate;
    float rotationEstimate;
    std::vector<std::pair<cv::KeyPoint, int> > finalActiveKeypoints;

    estimate(activeKeypoints, center, scaleEstimate, rotationEstimate,
             finalActiveKeypoints);

    // Update object state estimate
    m_center = center;
    m_scaleEstimate = scaleEstimate;
    m_rotationEstimate = rotationEstimate;
    m_activeKeypoints = finalActiveKeypoints;
    m_trackedKeypoints = trackedKeypoints;

    m_hasResult = false;
    if (!(m_center.x == NAN && m_center.y == NAN) &&
        (m_activeKeypoints.size() > (m_numInitialKeypoints / 10.0)))
    {
        m_hasResult = true;

        // Compute the bounding box of the object for the current frame
        m_resultBBtopLeft = m_center + m_scaleEstimate * rotate(m_centerToTopLeft,
                                                                m_rotationEstimate);
        m_resultBBtopRight = m_center + m_scaleEstimate * rotate(m_centerToTopRight,
                                                                 m_rotationEstimate);
        m_resultBBbottomLeft = m_center + m_scaleEstimate * rotate(m_centerToBottomLeft,
                                                                   m_rotationEstimate);
        m_resultBBbottomRight = m_center + m_scaleEstimate * rotate(m_centerToBottomRight,
                                                                    m_rotationEstimate);
    }
    else
    {
        m_hasResult = false;
        m_scaleEstimate = NAN;
        m_rotationEstimate = NAN;
        m_center = cv::Point2f(NAN, NAN);
    }

}

void CMT::estimate(const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIn,
                   cv::Point2f                                      &center,
                   float                                            &scaleEstimate,
                   float                                            &rotationEstimate,
                   std::vector<std::pair<cv::KeyPoint, int> >       &keypointsOut)
{
    center = cv::Point2f(NAN,NAN);
    scaleEstimate = NAN;
    rotationEstimate = NAN;

    //At least 2 keypoints are needed for scale
    if(keypointsIn.size() > 1)
    {
        keypointsOut = std::vector<std::pair<cv::KeyPoint, int> >();
        //sort
        std::vector<std::pair<int, int> > list;
        for(int i = 0; i < keypointsIn.size(); i++)
        {
            list.push_back(std::make_pair(keypointsIn[i].second, i));
        }

        std::sort(&list[0], &list[0]+list.size(), compareInt);
        for(int i = 0; i < list.size(); i++) {
            keypointsOut.push_back(keypointsIn[list[i].second]);
        }


        std::vector<int> ind1;
        std::vector<int> ind2;
        for(int i = 0; i < list.size(); i++)
            for(int j = 0; j < list.size(); j++)
            {
                if(i != j && keypointsOut[i].second != keypointsOut[j].second)
                {
                    ind1.push_back(i);
                    ind2.push_back(j);
                }
            }
        if(ind1.size() > 0)
        {
            std::vector<int> class_ind1;
            std::vector<int> class_ind2;
            std::vector<cv::KeyPoint> pts_ind1;
            std::vector<cv::KeyPoint> pts_ind2;
            for(int i = 0; i < ind1.size(); i++)
            {
                class_ind1.push_back(keypointsOut[ind1[i]].second - 1);
                class_ind2.push_back(keypointsOut[ind2[i]].second - 1);
                pts_ind1.push_back(keypointsOut[ind1[i]].first);
                pts_ind2.push_back(keypointsOut[ind2[i]].first);
            }
            std::vector<float> scaleChange;
            std::vector<float> angleDiffs;
            for(int i = 0; i < pts_ind1.size(); i++)
            {
                cv::Point2f p = pts_ind2[i].pt - pts_ind1[i].pt;
                //This distance might be 0 for some combinations,
                //as it can happen that there is more than one keypoint at a single location
                float dist = sqrt(p.dot(p));
                float origDist = m_squareFormDists[class_ind1[i]][class_ind2[i]];
                scaleChange.push_back(dist/origDist);
                //Compute angle
                float angle = atan2(p.y, p.x);
                float origAngle = m_squareFormAngles[class_ind1[i]][class_ind2[i]];
                float angleDiff = angle - origAngle;
                //Fix long way angles
                if(fabs(angleDiff) > CV_PI)
                {
                    int sign = angleDiff / (fabs(angleDiff));
                    angleDiff -= sign * 2 * CV_PI;
                }
                angleDiffs.push_back(angleDiff);
            }
            scaleEstimate = median(scaleChange);
            if(!m_estimateScale)
                scaleEstimate = 1;
            rotationEstimate = median(angleDiffs);
            if(!m_estimateRotation)
                rotationEstimate = 0;
            m_votes = std::vector<cv::Point2f>();
            for(unsigned int i = 0; i < keypointsOut.size(); i++)
                m_votes.push_back(keypointsOut[i].first.pt - scaleEstimate * rotate(
                                    m_springs[keypointsOut[i].second-1], rotationEstimate));

            // Cluster the votes using agglomerative hierarchical clustering
            // using the complete linkage method based on eucledian distance
            // with threshold m_threshOutlier
            AgglomerativeHierarchicalCluster ahc =
                    AgglomerativeHierarchicalCluster(m_votes, m_threshOutlier);

            std::vector<int> T;
            T = ahc.getClusters();

            //Get largest class
            int Cmax = ahc.getLargestCluster();

            //Remember outliers
            m_outliers = std::vector<std::pair<cv::KeyPoint, int> >();
            std::vector<std::pair<cv::KeyPoint, int> > newKeypoints;
            std::vector<cv::Point2f> newVotes;
            for(unsigned int i = 0; i < keypointsOut.size(); i++)
            {
                if(T[i] != Cmax)
                    m_outliers.push_back(keypointsOut[i]);
                else
                {
                    newKeypoints.push_back(keypointsOut[i]);
                    newVotes.push_back(m_votes[i]);
                }
            }
            keypointsOut = newKeypoints;

            if (keypointsOut.size() > 1)
            {
                // Recalculate scale and rotation
                //sort
                list.clear();
                for(int i = 0; i < keypointsOut.size(); i++)
                {
                    list.push_back(std::make_pair(keypointsOut[i].second, i));
                }

                std::sort(&list[0], &list[0]+list.size(), compareInt);

                ind1.clear();
                ind2.clear();
                for(int i = 0; i < list.size(); i++)
                    for(int j = 0; j < list.size(); j++)
                    {
                        if(i != j && keypointsOut[i].second != keypointsOut[j].second)
                        {
                            ind1.push_back(i);
                            ind2.push_back(j);
                        }
                    }
                if(ind1.size() > 0)
                {
                    std::vector<int> class_ind1;
                    std::vector<int> class_ind2;
                    std::vector<cv::KeyPoint> pts_ind1;
                    std::vector<cv::KeyPoint> pts_ind2;
                    for(int i = 0; i < ind1.size(); i++)
                    {
                        class_ind1.push_back(keypointsOut[ind1[i]].second - 1);
                        class_ind2.push_back(keypointsOut[ind2[i]].second - 1);
                        pts_ind1.push_back(keypointsOut[ind1[i]].first);
                        pts_ind2.push_back(keypointsOut[ind2[i]].first);
                    }
                    std::vector<float> scaleChange;
                    std::vector<float> angleDiffs;
                    for(int i = 0; i < pts_ind1.size(); i++)
                    {
                        cv::Point2f p = pts_ind2[i].pt - pts_ind1[i].pt;
                        //This distance might be 0 for some combinations,
                        //as it can happen that there is more than one keypoint at a single location
                        float dist = sqrt(p.dot(p));
                        float origDist = m_squareFormDists[class_ind1[i]][class_ind2[i]];
                        scaleChange.push_back(dist/origDist);
                        //Compute angle
                        float angle = atan2(p.y, p.x);
                        float origAngle = m_squareFormAngles[class_ind1[i]][class_ind2[i]];
                        float angleDiff = angle - origAngle;
                        //Fix long way angles
                        if(fabs(angleDiff) > CV_PI)
                        {
                            int sign = angleDiff / (fabs(angleDiff));
                            angleDiff -= sign * 2 * CV_PI;
                        }
                        angleDiffs.push_back(angleDiff);
                    }
                    scaleEstimate = median(scaleChange);
                    if(!m_estimateScale)
                        scaleEstimate = 1;
                    rotationEstimate = median(angleDiffs);
                    if(!m_estimateRotation)
                        rotationEstimate = 0;
                }
                center = cv::Point2f(0,0);
                for(unsigned int i = 0; i < newVotes.size(); i++)
                    center += newVotes[i];
                center *= (1.0/newVotes.size());
            }
        }
    }
}

cv::Point2f CMT::getCenter()
{
    return m_center;
}

float CMT::getScaleEstimate()
{
    return m_scaleEstimate;
}

float CMT::getRotationEstimate()
{
    return m_rotationEstimate;
}

int CMT::getNumberOfActiveKeypoints()
{
    return m_activeKeypoints.size();
}

void CMT::showAllInfo(cv::Mat &frame)
{
    if (m_hasResult)
    {
        cv::line(frame, m_resultBBbottomLeft, m_resultBBbottomRight,
                 CV_RGB(0, 0, 255), 2, 8);
        cv::line(frame, m_resultBBbottomRight, m_resultBBtopRight,
                 CV_RGB(0, 0, 255), 2, 8);
        cv::line(frame, m_resultBBtopRight, m_resultBBtopLeft,
                 CV_RGB(0, 0, 255), 2, 8);
        cv::line(frame, m_resultBBtopLeft, m_resultBBbottomLeft,
                 CV_RGB(0, 0, 255), 2, 8);
    }

    if (m_activeKeypoints.size() > 0)
    {
        std::vector<cv::KeyPoint> keypoints;
        for (int i = 0; i < m_activeKeypoints.size(); i++)
        {
            keypoints.push_back(m_activeKeypoints[i].first);
        }
        cv::drawKeypoints(frame, keypoints, frame, CV_RGB(0, 255, 0));
    }

    if (m_outliers.size() > 0)
    {
        std::vector<cv::KeyPoint> outliers;
        for (int i = 0; i < m_outliers.size(); i++)
        {
            outliers.push_back(m_outliers[i].first);
        }
        cv::drawKeypoints(frame, outliers, frame, CV_RGB(255, 0, 0));
    }
}
