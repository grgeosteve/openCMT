#include <cmath>
#include <opencv2/video/tracking.hpp>
#include "utils.h"

void CMT::inRect(const cv::Rect                  rect,
                 const std::vector<cv::KeyPoint> &keypoints,
                 std::vector<cv::KeyPoint>       &in,
                 std::vector<cv::KeyPoint>       &out)
{
    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::Point2f keypoint;
        keypoint = keypoints[i].pt;

        bool c1 = (keypoint.x > rect.tl().x) && (keypoint.x < rect.br().x);
        bool c2 = (keypoint.y > rect.tl().y) && (keypoint.y < rect.br().y);
        bool in_rectangle = c1 && c2;

        if (in_rectangle)
        {
            in.push_back(keypoints[i]);
        }
        else
        {
            out.push_back(keypoints[i]);
        }
    }
}

float computeDistance(cv::KeyPoint kp1,
                      cv::KeyPoint kp2)
{
    cv::Point2f point1 = kp1.pt;
    cv::Point2f point2 = kp2.pt;
    cv::Point2f d = point2 - point1;
    float distance = sqrt(d.dot(d));
    return distance;
}

float computeAngle(cv::KeyPoint kp1,
                   cv::KeyPoint kp2)
{
    cv::Point2f point1 = kp1.pt;
    cv::Point2f point2 = kp2.pt;

    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;

    float angle = atan2(dy, dx);
    return angle;
}

void computeSquareformDist(std::vector<cv::KeyPoint>        &keypoints,
                           std::vector<std::vector<float> > &squareformDist)
{
    for (int i = 0; i < keypoints.size(); i++)
    {
        std::vector<float> dists;
        for (int j = 0; j < keypoints.size(); j++)
        {
            if (i < j)
            {
                dists.push_back(computeDistance(keypoints[i], keypoints[j]));
            }
            else if (i > j)
            {
                dists.push_back(squareformDist[i][j]);
            }
            else
            {
                dists.push_back(0);
            }
        }
        squareformDist.push_back(dists);
    }
}

void computeSquareformAngle(std::vector<cv::KeyPoint>        &keypoints,
                            std::vector<std::vector<float> > &squareformAngles)
{
    for (int i = 0; i < keypoints.size(); i++)
    {
        std::vector<float> angles;
        for (int j = 0; j < keypoints.size(); j++)
        {
            angles.push_back(computeAngle(keypoints[i], keypoints[j]));
        }
        squareformAngles.push_back(angles);
    }
}

void trackLK(cv::Mat                                    imageGray,
             cv::Mat                                    previousImageGray,
             std::vector<std::pair<cv::KeyPoint, int> > &keypoints,
             std::vector<std::pair<cv::KeyPoint, int> > &trackedKeypoints,
             int                                        threshFB)
{
    // First check if there are any keypoints in the keypoints vector
    if (keypoints.size() == 0)
    {
        trackedKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();
        return;
    }

    // Extract the points from the input keypoints
    std::vector<cv::Point2f> points;
    for (int i = 0; i < keypoints.size(); i++)
    {
        points.push_back(keypoints[i].first.pt);
    }

    std::vector<cv::Point2f> nextPoints;
    std::vector<uchar>       fStatus;
    std::vector<float>       fError;
    // Calculate forward optical flow
    cv::calcOpticalFlowPyrLK(previousImageGray, imageGray, points, nextPoints, fStatus, fError);

    std::vector<cv::Point2f> prevPoints;
    std::vector<uchar>       bStatus;
    std::vector<float>       bError;
    // Calculate backward optical flow
    cv::calcOpticalFlowPyrLK(imageGray, previousImageGray, nextPoints, prevPoints, bStatus, bError);

    // Calculate forward-backward error
    std::vector<float> fbError;
    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::Point2f vec;
        vec = prevPoints[i] - points[i];
        fbError.push_back(sqrt(vec.dot(vec)));
    }

    // Set status of tracked keypoints depending on forward-backward
    // error and lukas-kanade error
    std::vector<uchar> status;
    for (int i = 0; i < keypoints.size(); i++)
    {
        status.push_back((fbError[i] <= threshFB) & fStatus[i]);
    }

    // Keep only the keypoints that are successfully tracked
    // and have relatively small forward-backward error
    trackedKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();
    for (int i = 0; i < keypoints.size(); i++)
    {
        if (status[i])
        {
            std::pair<cv::KeyPoint, int> point = keypoints[i];
            point.first.pt = nextPoints[i];
            trackedKeypoints.push_back(point);
        }
    }
}
