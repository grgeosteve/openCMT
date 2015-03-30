#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class AgglomerativeHierarchicalCluster
{
public:
    AgglomerativeHierarchicalCluster(const std::vector<cv::Point2f> &votes,
                                     int                            thresh);
    std::vector<int> getClusters();
    int getLargestCluster();
    int getLargestClusterSize();

    std::vector<std::vector<float> > squareformDists;

private:
    void initializeClusters();
    void computeClusters(int thresh);
    float computeLinkage(int clusterIndex1,
                         int clusterIndex2);
    void computeLargestCluster();

    int h_numPoints;
    std::vector<int> h_pointIndex;
    std::vector<int> h_clusterIndex;
    std::map<int, std::vector<int> > clusters;
    int h_numClusters;
    int h_largestClusterSize;
    int h_largestClusterIndex;
};

#endif // CLUSTER_H
