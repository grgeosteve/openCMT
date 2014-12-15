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

        bool c1 = (keypoint.x > rect.tl.x) && (keypoint.x < rect.br.x);
        bool c2 = (keypoint.y > rect.tl.y) && (keypoint.y < rect.br.y);
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
