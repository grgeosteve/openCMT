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

}

#endif // UTILS_H
