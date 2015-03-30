#include <cmath>
#include <opencv2/video/tracking.hpp>
#include <assert.h>
#include "utils.h"

void inRect(const cv::Rect                  &rect,
            const std::vector<cv::KeyPoint> &keypoints,
            std::vector<cv::KeyPoint>       &in,
            std::vector<cv::KeyPoint>       &out)
{
    assert(keypoints.size() > 0);

    in = std::vector<cv::KeyPoint>();
    out = std::vector<cv::KeyPoint>();
    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::Point2f keypoint;
        keypoint = keypoints[i].pt;

        bool c1 = (keypoint.x > rect.tl().x) && (keypoint.x < rect.br().x);
        bool c2 = (keypoint.y > rect.tl().y) && (keypoint.y < rect.br().y);
        bool in_rectangle = c1 && c2;

        if (in_rectangle)
            in.push_back(keypoints[i]);
        else
            out.push_back(keypoints[i]);
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

float computeDistance(cv::Point2f p1,
                      cv::Point2f p2)
{
    cv::Point2f d = p2 - p1;
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

void computeSquareformDist(const std::vector<cv::KeyPoint>  &keypoints,
                           std::vector<std::vector<float> > &squareformDist)
{
    assert(keypoints.size() > 0);

    squareformDist = std::vector<std::vector<float> >();
    for (int i = 0; i < keypoints.size(); i++)
    {
        squareformDist.push_back(std::vector<float>());
        for (int j = 0; j < keypoints.size(); j++)
        {
            squareformDist[i].push_back(0);
        }
    }

    for (int i = 0;i < keypoints.size(); i++)
    {
        for (int j = 0; j < keypoints.size(); j++)
        {
            if (i < j)
            {
                squareformDist[i][j] = computeDistance(keypoints[i], keypoints[j]);
            }
            else if (i > j)
            {
                squareformDist[i][j] = squareformDist[j][i];
            }
            else
            {
                squareformDist[i][j] = 0;
            }
        }
    }
}

void computeSquareformDist(const std::vector<cv::Point2f>   &points,
                           std::vector<std::vector<float> > &squareformDist)
{
    assert(points.size() > 0);

    squareformDist = std::vector<std::vector<float> >();
    for (int i = 0; i < points.size(); i++)
    {
        squareformDist.push_back(std::vector<float>());
        for (int j = 0; j < points.size(); j++)
        {
            squareformDist[i].push_back(0);
        }
    }
    for (int i = 0; i < points.size(); i++)
    {
        for (int j = 0; j < points.size(); j++)
        {
            if (i < j)
            {
                squareformDist[i][j] = computeDistance(points[i], points[j]);
            }
            else if (j < i)
            {
                squareformDist[i][j] = squareformDist[j][i];
            }
            else
            {
                squareformDist[i][j] = 0;
            }
        }
    }
}

void computeSquareformAngle(const std::vector<cv::KeyPoint>     &keypoints,
                            std::vector<std::vector<float> >    &squareformAngles)
{
    assert(keypoints.size() > 0);

    squareformAngles = std::vector<std::vector<float> >();
    for (int i = 0; i < keypoints.size(); i++)
    {
        std::vector<float> angles;
        for(int j = 0; j < keypoints.size(); j++)
        {
            angles.push_back(computeAngle(keypoints[i], keypoints[j]));
        }
        squareformAngles.push_back(angles);
    }
}

std::vector<bool> inVector(const std::vector<int> &a,
                           const std::vector<int> &b)
{
    assert(a.size() > 0);
    assert(b.size() > 0);

    std::vector<bool> result(a.size());

    for (int i = 0; i < a.size(); i++)
    {
        if (std::find(b.begin(), b.end(), a[i]) != b.end())
        {
            result[i] = true;
        }
        else
        {
            result[i] = false;
        }
    }
    return result;
}

bool compareInt(const std::pair<int, int> &left,
                const std::pair<int, int> &right)
{
    return left.first > right.first;
}

float median(std::vector<float> vec)
{
    assert(vec.size() > 0);

    std::sort(vec.begin(), vec.end());

    if (vec.size() % 2 == 0)
    {
        int index1 = (vec.size() / 2) - 1;
        int index2 = vec.size() / 2;

        return (vec[index1] + vec[index2]) / 2.0;
    }
    else
    {
        int index = (vec.size() / 2) + 1;
        return vec[index];
    }
}

cv::Point2f rotate(const cv::Point2f    &pt,
                   float                rad)
{
    float rotated_x;
    float rotated_y;

    rotated_x = pt.x * cos(rad) - pt.y * sin(rad);
    rotated_y = pt.x * sin(rad) + pt.y * cos(rad);

    return cv::Point2f(rotated_x, rotated_y);
}

void trackLK(const cv::Mat                                      &imageGray,
             const cv::Mat                                      &previousImageGray,
             const std::vector<std::pair<cv::KeyPoint, int> >   &keypoints,
             std::vector<std::pair<cv::KeyPoint, int> >         &trackedKeypoints,
             int                                                threshFB)
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
    cv::calcOpticalFlowPyrLK(previousImageGray,
                             imageGray,
                             points,
                             nextPoints,
                             fStatus,
                             fError);

    std::vector<cv::Point2f> prevPoints;
    std::vector<uchar>       bStatus;
    std::vector<float>       bError;
    // Calculate backward optical flow
    cv::calcOpticalFlowPyrLK(imageGray,
                             previousImageGray,
                             nextPoints,
                             prevPoints,
                             bStatus,
                             bError);

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
