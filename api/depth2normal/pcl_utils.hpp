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

static std::string pcd_path = "/home/alex/Projects/TRAIL/datasets/ExampleData/";

void compute_normals_integral_images()
{
    // load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (pcd_path, *cloud);

    // estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    // visualize normals
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor (0.0, 0.0, 0.5);
    // viewer.addPointCloud<pcl::PointXYZ>(cloud);
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals);

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}

pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr compute_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                pcl::PointCloud<pcl::Normal>::Ptr normals,
                                                                pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures)
{
    // Setup the principal curvatures computation
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    // Provide the original point cloud (without normals)
    principal_curvatures_estimation.setInputCloud (cloud);

    // Provide the point cloud with normals
    principal_curvatures_estimation.setInputNormals (normals);

    // Use the same KdTree from the normal estimation
    principal_curvatures_estimation.setSearchMethod (tree);
    principal_curvatures_estimation.setRadiusSearch(1.0);
    // principal_curvatures_estimation.setKSearch (10);

    // Actually compute the principal curvatures
    principal_curvatures_estimation.compute (*principal_curvatures);

    return principal_curvatures;
}

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

    // pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
    // pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    tree->setInputCloud(cloud);
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);

    //ne.setRadiusSearch (0.5);
    ne.setKSearch(10);

    // Set the view point
    // ne.setViewPoint (std::numeric_limits<float>::max (), 
    //                 std::numeric_limits<float>::max (), 
    //                 std::numeric_limits<float>::max ());

    ne.setViewPoint(0.0, 0.0, 0.0);    
    float sensor_vpx, sensor_vpy, sensor_vpz;
    ne.getViewPoint(sensor_vpx, sensor_vpy, sensor_vpz);
    std::cout << "Sensor viewpoint is: " << sensor_vpx << ", " << sensor_vpy << ", " << sensor_vpz << endl; 

    std::cout << "computing normals...\n";
    // Compute the features
    ne.compute(*cloud_normals);
    std::cout << "cloud_normals->size (): " << cloud_normals->size () << std::endl;

    pcl::concatenateFields (*cloud, *cloud_normals, *cloud_with_normals);

    int total_pixels_negative_zero_nz = 0; 
    for(int nIndex = 0; nIndex < cloud->points.size (); nIndex += 5000)
    {
        if (!isnan(cloud->points[nIndex].x) && !isnan(cloud->points[nIndex].y) && !isnan(cloud->points[nIndex].z)) 
        {
            cout << "XYZ: " << cloud->points[nIndex].x << ", " << cloud->points[nIndex].y << ", " << cloud->points[nIndex].z << endl;
        }
        if (!isnan(cloud_with_normals->points[nIndex].normal_x) && !isnan(cloud_with_normals->points[nIndex].normal_y) && !isnan(cloud_with_normals->points[nIndex].normal_z))
        {
            cout << "Normals: " << cloud_with_normals->points[nIndex].normal_x << ", " << cloud_with_normals->points[nIndex].normal_y << ", " << cloud_with_normals->points[nIndex].normal_z << endl;
        }
    }

    for(int nIndex = 0; nIndex < cloud->points.size (); nIndex++)
    {
        if (cloud_with_normals->points[nIndex].normal_z <= 0.0 || isnan(cloud_with_normals->points[nIndex].normal_z)) 
        {
            total_pixels_negative_zero_nz++;
        }
    }

    if (cloud_with_normals->points.size() != cloud->points.size()) 
    {
        cout << "GRAVE MISTAKE: NUMBER OF POINTS NOT EQUAL!!!!!!!!!!!!!!!!!!" << endl;
    }
    cout << "Negative or zero nz pixels: " << total_pixels_negative_zero_nz << " out of total pixels: " << cloud->points.size() << endl;

    // std::vector<int> indices;
    // pcl::removeNaNNormalsFromPointCloud(*cloud_with_normals, *cloud_with_normals, indices);
    // std::cout << "filtered final output->size (): " << cloud_with_normals->size () << std::endl;
    // principal_curvatures = compute_curvature(cloud, cloud_normals, principal_curvatures);
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ>);
    // // Provide the original point cloud (without normals)
    // principal_curvatures_estimation.setInputCloud (cloud);
    // // Provide the point cloud with normals
    // principal_curvatures_estimation.setInputNormals (cloud_normals);
    // // Use the same KdTree from the normal estimation
    // principal_curvatures_estimation.setSearchMethod (tree2);
    // principal_curvatures_estimation.setRadiusSearch(1.0);
    // // principal_curvatures_estimation.setKSearch (10);
    // // Actually compute the principal curvatures
    // principal_curvatures_estimation.compute (*principal_curvatures);

    std::stringstream ss1, ss2;
    ss1 << pcd_path << file.substr(0, file.size()-4) << "_normals.pcd";
    ss2 << pcd_path << file.substr(0, file.size()-4) << "_concat.pcd";
    // writer.write<pcl::Normal>(ss1.str(), *cloud_normals, true);
    // writer.write<pcl::PointNormal>(ss2.str(), *cloud_with_normals, true);
    pcl::io::savePCDFileBinary(ss2.str(), *cloud_with_normals);

    // visualize normals
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor (0.0, 0.0, 0.5);
    // viewer->addPointCloud<pcl::PointNormal>(cloud_with_normals);
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, level, visualize_scale, "Normals");
    viewer->addPointCloud<pcl::PointXYZ> (camera);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
    // viewer->addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, principal_curvatures, 100, 0.01, "cloud_curvatures");

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce ();
    }
}

void display_point_cloud(std::string file, std::string type)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // visualize normals
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor (0.0, 0.0, 0.5);

    pcd_path += file;

    pcl::io::loadPCDFile (pcd_path, *cloud);

    // load point cloud
    if (type == "xyz")
    {
        viewer.addPointCloud<pcl::PointXYZ>(cloud);
    }
    else if (type == "normals")
    {
        pcl::io::loadPCDFile (pcd_path, *cloud_normals);
        viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, 10, 1);
    }

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
}