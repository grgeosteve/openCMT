#include <cmath>
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
