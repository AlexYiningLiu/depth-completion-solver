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
const string depth_img_path = "/home/alex/Projects/TRAIL/depth-completion-solver/input_data/view_71_raw.png";

static void Threshold_Demo(int, void*)
{
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
    */
    Mat src, dst;
    src = imread(depth_img_path, IMREAD_ANYDEPTH);

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

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        cout << endl;
        cout << "NEED 3 INPUT ARGUMENTS" << endl;
        cout << endl;
        cout << "DEPTH2NORMALS" << endl;
        cout << "Compute the normals of a grayscale/depth image." << endl;
        cout << endl;
        cout << "Usage:" << endl;
        cout <<"\t./depth2normals FILE LEVEL VISUALIZATION_SCALE" << endl;
        cout << endl;
        cout << "FILE: grayscale/depth image path with extension" << endl;
        cout << "LEVEL: integer that controls how many points to skip in visualization" << endl;
        cout << "VISUALIZATION_SCALE: float that controls scale of surface normals visualization" << endl;
        cout << endl;
        cout << endl;
    }
    
    string FILE(argv[1]);
    int LEVEL(atoi(argv[2]));
    float VISUALIZATION_SCALE(atof(argv[3]));

    compute_normals_pca(FILE, LEVEL, VISUALIZATION_SCALE);

    return 0;
}
