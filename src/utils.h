#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

namespace CMT
{
    /*
     * inRect: takes as input a rectangle 'rect', and a set of keypoints 'keypoints'
     * and returns two vectors of keypoints, 'in' which lie in the rectangle,
     * and 'out' which lie outside of the rectangle.
     */
    void inRect(const cv::Rect                  rect,
                const std::vector<cv::KeyPoint> &keypoints,
                std::vector<cv::KeyPoint>       &in,
                std::vector<cv::KeyPoint>       &out);

    /*
     * computeDistance: computes the Euclidean distance between two
     * keypoints
     */
    float computeDistance(cv::KeyPoint kp1,
                          cv::KeyPoint kp2);

    /*
     * computeAngle: computes the angle betwen two keypoints
     */
    float computeAngle(cv::KeyPoint kp1,
                       cv::KeyPoint kp2);

    /*
     * computeSquareformDist: computes a squareform distance table
     */
    void computeSquareformDist(std::vector<cv::KeyPoint>        &keypoints,
                               std::vector<std::vector<float> > &squareformDist);

    /*
     * computeSquareformAngle: computes a squareform angle table
     */
    void computeSquareformAngle(std::vector<cv::KeyPoint>        &keypoints,
                                std::vector<std::vector<float> > &squareformAngles);

    /*
     * trackLK: tracks keypoints between frames using
     * Lukas-Kanade pyramidal optical flow method
     */
    void trackLK(cv::Mat                                    imageGray,
                 cv::Mat                                    previousImageGray,
                 std::vector<std::pair<cv::KeyPoint, int> > &keypoints,
                 std::vector<std::pair<cv::KeyPoint, int> > &trackedKeypoints,
                 int                                        threshFB=20);
}

#endif // UTILS_H
