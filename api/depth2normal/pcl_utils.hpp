#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

#include <math.h>
#include <iostream>
#include <string>

static std::string pcd_path = "/home/alex/Projects/TRAIL/depth-completion-solver/input_data/";
static std::string out_path = "/home/alex/Projects/TRAIL/depth-completion-solver/temporary_data/";

void compute_normals_pca(std::string file, int level, float visualize_scale)
{
    std::string curr_pcd_path = pcd_path + file;

    // load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (curr_pcd_path, *cloud);
    pcl::PCDWriter writer;
    std::cout << "point cloud->size(): " << cloud->size () << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr camera (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ point;
    point.x = 0.0;
    point.y = 0.0;
    point.z = 0.0;
    camera->points.push_back(point);

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    tree->setInputCloud(cloud);
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr only_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);

    //ne.setRadiusSearch (0.5);
    ne.setKSearch(10);

    // Set the view point
    ne.setViewPoint(0.0, 0.0, 0.0);    
    float sensor_vpx, sensor_vpy, sensor_vpz;
    ne.getViewPoint(sensor_vpx, sensor_vpy, sensor_vpz);
    std::cout << "Sensor viewpoint is: " << sensor_vpx << ", " << sensor_vpy << ", " << sensor_vpz << endl; 

    std::cout << "computing normals...\n";
    // Compute the features
    ne.compute(*only_normals);
    std::cout << "only_normals->size (): " << only_normals->size () << std::endl;

    pcl::concatenateFields (*cloud, *only_normals, *cloud_with_normals);
    if (cloud_with_normals->points.size() != cloud->points.size()) 
    {
        std::cout << "FATAL MISTAKE: NUMBER OF POINTS NOT EQUAL!" << std::endl;
    }

    std::stringstream ss1;
    ss1 << out_path << file.substr(0, file.size()-4) << "_concat.pcd";
    writer.write<pcl::PointNormal>(ss1.str(), *cloud_with_normals, true);
    pcl::io::savePCDFileBinary(ss1.str(), *cloud_with_normals);
    std::cout << "PCD file saved successfully at: " << ss1.str() << std::endl;

    // visualize normals
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor (0.0, 0.0, 0.5);
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, only_normals, level, visualize_scale, "Normals");
    viewer->addPointCloud<pcl::PointXYZ> (camera);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce ();
    }
}

void display_point_cloud(std::string file)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // visualization paramters
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor (0.0, 0.0, 0.5);

    pcd_path += file;

    // load point cloud
    pcl::io::loadPCDFile (pcd_path, *cloud);
    viewer.addPointCloud<pcl::PointXYZ>(cloud);

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
}