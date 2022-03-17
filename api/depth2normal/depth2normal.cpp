#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "pcl_utils.hpp"

using namespace cv;
using namespace std;

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 16000;
int const max_type = 4;
int const max_binary_value = 255;
const char* window_name = "Threshold Demo";
const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

static void Threshold_Demo(int, void*)
{
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
    */
    Mat src, dst;
    src = imread("view_2.png", IMREAD_ANYDEPTH);

    threshold(src, dst, threshold_value, max_binary_value, threshold_type );
    imshow( window_name, dst );
}

int compute_nonzero_mask()
{
    namedWindow( window_name, WINDOW_AUTOSIZE ); // Create a window to display results
    createTrackbar( trackbar_type,
                    window_name, &threshold_type,
                    max_type, Threshold_Demo ); // Create a Trackbar to choose type of Threshold
    createTrackbar( trackbar_value,
                    window_name, &threshold_value,
                    max_value, Threshold_Demo ); // Create a Trackbar to choose Threshold value
    Threshold_Demo( 0, 0 ); // Call the function to initialize
    waitKey();
    return 0;
}

void compute_normals_simple(string FILE)
{
    Mat depth = imread(FILE, IMREAD_ANYDEPTH);
    depth.convertTo(depth, CV_64FC3);
    Mat nor(depth.size(), CV_64FC3);

    for(int x = 1; x < depth.cols - 1; ++x)
    {
        for(int y = 1; y < depth.rows - 1; ++y)
        {
            Vec3d t(x, y-1, depth.at<double>(y-1, x)/*depth(y-1,x)*/);  // take coordinate vector of top neighbor
            Vec3d l(x-1, y ,depth.at<double>(y, x-1)/*depth(y,x-1)*/);  // take coordinate vector of left neighbor
            Vec3d c(x, y, depth.at<double>(y, x)/*depth(y,x)*/);  // take coordinate vector at current position
            Vec3d d = (l-c).cross(t-c);  // {-1, 0, delta_depth_left} X {0, -1, delta_depth_top}
            Vec3d n = normalize(d);
            nor.at<Vec3d>(y, x) = n;
        }
    }

    imshow("normals", nor);
    waitKey(0);
}

// https://mcarletti.github.io/articles/depth2normals/
void compute_normals_neighbor_cross(string FILE, string MODE)
{
    // load image as grayscale and convert to 32 bits (single channel)
    Mat depth = imread(FILE, IMREAD_ANYDEPTH);
    // cvtColor(depth, depth, CV_RGB2GRAY);
    depth.convertTo(depth, CV_32FC1);

    if (MODE == "conv")
    {
        // Shape, Illumination, and Reflectance from Shading
        // Jonathan T. Barron (https://jonbarron.info/), Jitendra Malik 
        // IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2015 
        // https://drive.google.com/file/d/1RvyCiDMg--jyO8lLBvopp0o271LvREoa/view

        // filters
        Mat_<float> f1 = (Mat_<float>(3, 3) << 1,  2,  1,
                                                0,  0,  0,
                                                -1, -2, -1) / 8;

        Mat_<float> f2 = (Mat_<float>(3, 3) << 1, 0, -1,
                                                2, 0, -2,
                                                1, 0, -1) / 8;

        /* Other filters that could be used:
        % f1 = [0, 0, 0;
        %       0, 1, 1;
        %       0,-1,-1]/4;
        % 
        % f2 = [0, 0, 0;
        %       0, 1,-1;
        %       0, 1,-1]/4;

        or

        % f1 = [0, 1, 0;
        %       0, 0, 0;
        %       0,-1, 0]/2;
        % 
        % f2 = [0, 0, 0;
        %       1, 0, -1;
        %       0, 0, 0]/2;
        */


        Mat f1m, f2m;
        flip(f1, f1m, 0);
        flip(f2, f2m, 1);

        Mat n1, n2;
        filter2D(depth, n1, -1, f1m, Point(-1, -1), 0, BORDER_CONSTANT);
        filter2D(depth, n2, -1, f2m, Point(-1, -1), 0, BORDER_CONSTANT);

        n1 *= -1;
        n2 *= -1;

        Mat temp = n1.mul(n1) + n2.mul(n2) + 1;
        cv::sqrt(temp, temp);

        Mat N3 = 1 / temp;
        Mat N1 = n1.mul(N3);
        Mat N2 = n2.mul(N3);

        vector<Mat> N;
        N.push_back(N1);
        N.push_back(N2);
        N.push_back(N3);

        Mat normals;
        merge(N, normals);

        imshow("convolution_based_normals", normals);
        normals *= 255;
        normals.convertTo(normals, CV_8UC3);
        imwrite("conv_normals.png", normals);
    }
    else if (MODE == "cross")
    {
        Mat normals(depth.size(), CV_32FC3);

        for(int x = 0; x < depth.cols; ++x)
        {
            for(int y = 0; y < depth.rows; ++y)
            {
                    // 3d pixels, think (x,y, depth)
                    /* * * * *
                    * * t * *
                    * l c * *
                    * * * * */

                Vec3f t(x,y-1,depth.at<float>(y-1, x));
                Vec3f l(x-1,y,depth.at<float>(y, x-1));
                Vec3f c(x,y,depth.at<float>(y, x));

                Vec3f d = (l-c).cross(t-c);

                Vec3f n = normalize(d);
                normals.at<Vec3f>(y,x) = n;
            }
        }

        imshow("explicitly cross_product normals", normals);
        normals *= 255;
        normals.convertTo(normals, CV_8UC3);
        imwrite("cross_normals.png", normals);
    }   
    else if (MODE == "crossfast")
    {
        Mat normals(depth.size(), CV_32FC3);

        for(int x = 0; x < depth.rows; ++x)
        {
            for(int y = 0; y < depth.cols; ++y)
            {
                // use float instead of double otherwise you will not get the correct result
                // check my updates in the original post. I have not figure out yet why this
                // is happening.
                float dzdx = (depth.at<float>(x+1, y) - depth.at<float>(x-1, y)) / 2.0;
                float dzdy = (depth.at<float>(x, y+1) - depth.at<float>(x, y-1)) / 2.0;

                Vec3f d(-dzdx, -dzdy, 1.0f);

                Vec3f n = normalize(d);
                normals.at<Vec3f>(x, y) = n;
            }
        }

        imshow("fast cross_product normals", normals);
        normals *= 255;
        normals.convertTo(normals, CV_8UC3);
        imwrite("crossfast_normals.png", normals);
    }
    else if (MODE == "jetmap")
    {
        // Multimodal Deep Learning for Robust RGB-D Object Recognition
        // Andreas Eitel, Jost Tobias Springenberg, Luciano Spinello, Martin Riedmiller, Wolfram Burgard
        // https://ieeexplore.ieee.org/abstract/document/7353446

        depth.convertTo(depth, CV_8UC1);
        Mat embedding(depth.size(), CV_8UC3);

        applyColorMap(depth, embedding, COLORMAP_JET);

        imshow("depth information to jet colormap", embedding);
        imwrite("jetmapembedding.png", embedding);
    } 

    waitKey(0);
}

int main(int argc, char* argv[])
{
    // // check input
    // if (argc != 3)
    // {
    //     cout << endl;
    //     cout << "DEPTH2NORMALS" << endl;
    //     cout << "Compute the normals of a grayscale/depth image." << endl;
    //     cout << endl;
    //     cout << "Usage:" << endl;
    //     cout <<"\t./depth2normals FILE MODE" << endl;
    //     cout << endl;
    //     cout << "FILE: grayscale/depth image path with extension" << endl;
    //     cout << "MODE: method to compute the normals; must be in [conv, cross, crossfast, jetmap]" << endl;
    //     cout << endl;

    //     return -1;
    // }

    if (argc != 4)
    {
        cout << endl;
        cout << "NEED 3 INPUT ARGUMENTS" << endl;
        cout << endl;
    }
    
    string FILE(argv[1]);
    int LEVEL(atoi(argv[2]));
    float VISUALIZATION_SCALE(atof(argv[3]));

    // compute_nonzero_mask();
    // compute_normals_neighbor_cross(FILE, MODE);
    // compute_normals_simple(FILE);
    // compute_normals_integral_images();
    compute_normals_pca(FILE, LEVEL, VISUALIZATION_SCALE);
    // display_point_cloud(FILE, MODE);

    return 0;
}
