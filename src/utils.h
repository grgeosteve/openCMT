#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <algorithm>
#include <iterator>

/*
 * inRect: takes as input a rectangle 'rect', and a set of keypoints 'keypoints'
 * and return two vectors of keypoints, 'in' which lie in the rectangle,
 * and 'out' which lie outside of the rectangle.
 */
void inRect(const cv::Rect                    &rect,
            const std::vector<cv::KeyPoint>   &keypoints,
            std::vector<cv::KeyPoint>         &in,
            std::vector<cv::KeyPoint>         &out);

/*
 * computeDistance: computes the Euclidean distance between two
 * keypoints (points)
 */
float computeDistance(cv::KeyPoint kp1,
                      cv::KeyPoint kp2);
float computeDistance(cv::Point2f p1,
                      cv::Point2f p2);

/*
 * computeAngle: computes the angle between two keypoints
 */
float computeAngle(cv::KeyPoint kp1,
                   cv::KeyPoint kp2);

/*
 * computeSquareformDist: computes a squareform distance table
 */
void computeSquareformDist(const std::vector<cv::KeyPoint>  &keypoints,
                           std::vector<std::vector<float> > &squareformDist);

void computeSquareformDist(const std::vector<cv::Point2f>   &points,
                           std::vector<std::vector<float> > &squareformDist);

/*
 * computeSquareformAngle: computes a squareform angle table
 */
void computeSquareformAngle(const std::vector<cv::KeyPoint>     &keypoints,
                            std::vector<std::vector<float> >    &squareformAngles);

/*
 * inVector: compares the contents of 2 vectors
 * return a boolean vector with the same size as the a,
 * and shows if a value in a is found in b
 */
std::vector<bool> inVector(const std::vector<int> &a,
                           const std::vector<int> &b);

/*
 * compareInt: a custom compare function for use in sorting algorithms
 *
 * Uses pairs of integers as input
 */
bool compareInt(const std::pair<int, int> &left,
                const std::pair<int, int> &right);

/*
 * median: compute the median value of a vector of floats
 */
float median(std::vector<float> vec);

/*
 * rotate: calculate the rotated coordinates of a point in reference
 * to the center
 */
cv::Point2f rotate(const cv::Point2f &pt,
                   float             rad);

/*
 * trackLK: tracks keypoints between frames using
 * Lukas-Kanade pyramidal optical flow method
 */
void trackLK(const cv::Mat                                      &imageGray,
             const cv::Mat                                      &previousImageGray,
             const std::vector<std::pair<cv::KeyPoint, int> >   &keypoints,
             std::vector<std::pair<cv::KeyPoint, int> >         &trackedKeypoints,
             int                                                threshFB=20);

#endif // UTILS_H
