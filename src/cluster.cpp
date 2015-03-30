#include "cluster.h"
#include "utils.h"

AgglomerativeHierarchicalCluster::AgglomerativeHierarchicalCluster(const std::vector<cv::Point2f> &votes,
                                                                   int                            thresh)
{
    h_numPoints = votes.size();

    // Compute initial distances between the votes (initial linkages)
    squareformDists = std::vector<std::vector<float> >();
    computeSquareformDist(votes, squareformDists);

    h_pointIndex = std::vector<int>(h_numPoints);
    for (int i = 0; i < h_numPoints; i++)
    {
        h_pointIndex[i] = i;
    }
    initializeClusters();
    computeClusters(thresh);
}

void AgglomerativeHierarchicalCluster::initializeClusters()
{
    // Begin with the clustering assigning each vote-point to 1 cluster
    h_numClusters = h_numPoints;
    h_clusterIndex = std::vector<int>(h_numClusters);
    for (int i = 0; i < h_numClusters; i++)
    {
        h_clusterIndex[i] = i;
    }

    clusters = std::map<int, std::vector<int> >();
    for (int i = 0; i < h_numClusters; i++)
    {
        clusters[h_clusterIndex[i]] = std::vector<int>(1, h_pointIndex[i]);
    }
}

float AgglomerativeHierarchicalCluster::computeLinkage(int clusterIndex1,
                                                       int clusterIndex2)
{
    int cluster1_size = clusters[clusterIndex1].size();
    int cluster2_size = clusters[clusterIndex2].size();

    // Compute all distances between each combination of points
    // between cluster 1 and cluster 2
    std::vector<std::vector<float> > dists;
    for (int i = 0; i < cluster1_size; i++)
    {
        std::vector<float> tmpDistVec;
        for (int j = 0; j < cluster2_size; j++)
        {
            float dist = squareformDists[clusters[clusterIndex1][i]]
                                        [clusters[clusterIndex2][j]];
            tmpDistVec.push_back(dist);
        }
        dists.push_back(tmpDistVec);
    }

    float maxDist = 0;
    for (int i = 0; i < dists.size(); i++)
    {
        for (int j = 0; j < dists[i].size(); j++)
        {
            if (maxDist < dists[i][j])
            {
                maxDist = dists[i][j];
            }
        }
    }

    return maxDist;
}

void AgglomerativeHierarchicalCluster::computeClusters(int thresh)
{
    float minLinkage = 0;
    while ((minLinkage < thresh) && (h_numClusters > 1))
    {
        // The dictionary will not contain linkages between the same cluster
        int numLinkages = (h_numClusters - 1) * h_numClusters;

        // Create a dictionary of linkages between clusters
        std::vector<std::vector<float> > linkages(numLinkages);
        for (int i = 0; i < numLinkages; i++)
        {
            linkages[i] = std::vector<float>(3);
        }

        std::vector<std::vector<int> > linkagesClusterIndex(numLinkages);
        for (int i = 0; i < numLinkages; i++)
        {
            linkagesClusterIndex[i] = std::vector<int>(2);
        }

        int allLinkagesIndex = 0;
        for (int i = 0; i < h_numClusters; i++)
        {
            for (int j = 0; j < h_numClusters; j++)
            {
                if (i == j)
                {
                    continue;
                }
                linkages[allLinkagesIndex][0] = i;
                linkages[allLinkagesIndex][1] = j;

                linkagesClusterIndex[allLinkagesIndex][0] = h_clusterIndex[i];
                linkagesClusterIndex[allLinkagesIndex][1] = h_clusterIndex[j];

                if (i > j)
                {
                    int oppositeLinkageIndex = (h_numClusters - 1) * j + (i - 1);

                    linkages[allLinkagesIndex][2] = linkages[oppositeLinkageIndex][2];
                }
                else
                {
                    linkages[allLinkagesIndex][2] = computeLinkage(
                                linkagesClusterIndex[allLinkagesIndex][0],
                                linkagesClusterIndex[allLinkagesIndex][1]);
                }

                allLinkagesIndex++;
            }
        }

        // Find the minimum linkage and save it as minLinkage
        // and keep the clusters to merge if minLinkages < thresh
        minLinkage = linkages[0][2];
        int minLinkageIndex_i = static_cast<int>(linkages[0][0]);
        int minLinkageIndex_j = static_cast<int>(linkages[0][1]);

        for (int i = 1; i < numLinkages; i++)
        {
            if (minLinkage > linkages[i][2])
            {
                minLinkage = linkages[i][2];
                minLinkageIndex_i = static_cast<int>(linkages[i][0]);
                minLinkageIndex_j = static_cast<int>(linkages[i][1]);
            }
        }

        if (minLinkage <= thresh)
        {
            // Merge the clusters with the minLinkage
            int cluster1_index = h_clusterIndex[minLinkageIndex_i];
            int cluster2_index = h_clusterIndex[minLinkageIndex_j];

            int cluster2_size = clusters[cluster2_index].size();
            for (int i = 0; i < cluster2_size; i++)
            {
                clusters[cluster1_index].push_back(clusters[cluster2_index][i]);
            }

            // Find cluster2_index in the h_clusterIndex vector and remove it
            std::vector<int>::iterator index = std::find(h_clusterIndex.begin(),
                                                         h_clusterIndex.end(),
                                                         cluster2_index);

            h_clusterIndex.erase(index);

            // Remove cluster2 from clusters dictionary
            clusters.erase(cluster2_index);

            // Reduce h_numClusters by 1
            h_numClusters--;
        }
    }
    computeLargestCluster();
}

void AgglomerativeHierarchicalCluster::computeLargestCluster()
{
    int maxIndex = h_clusterIndex[0];
    int maxSize = clusters[maxIndex].size();

    if (h_numClusters > 1)
    {
        for (int i = 1; i < h_numClusters; i++)
        {
            int index = h_clusterIndex[i];
            int clusterSize = clusters[index].size();

            if (maxSize < clusterSize)
            {
                maxSize = clusterSize;
                maxIndex = index;
            }
        }

        h_largestClusterIndex = maxIndex;
        h_largestClusterSize = maxSize;
    }
    else
    {
        h_largestClusterIndex = maxIndex;
        h_largestClusterSize = maxSize;
    }
}

std::vector<int> AgglomerativeHierarchicalCluster::getClusters()
{
    std::vector<int> pointClusters(h_numPoints);
    for (int i = 0; i < h_numClusters; i++)
    {
        int clusterIndex = h_clusterIndex[i];

        int clusterSize = clusters[clusterIndex].size();
        for (int j = 0; j < clusterSize; j++)
        {
            int pointIndex = clusters[clusterIndex][j];
            pointClusters[pointIndex] = clusterIndex;
        }
    }

    return pointClusters;
}

int AgglomerativeHierarchicalCluster::getLargestCluster()
{
    return h_largestClusterIndex;
}

int AgglomerativeHierarchicalCluster::getLargestClusterSize()
{
    return h_largestClusterSize;
}
