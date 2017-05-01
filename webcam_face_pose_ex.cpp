// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

using namespace dlib;
using namespace std;

typedef cv::Matx44d HeadPose;

const static cv::Point3f P3D_SELLION(0., 0.,0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -65.5,-5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 65.5,-5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -77.5,-6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 77.5,-6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -48.0);
const static cv::Point3f P3D_STOMMION(10.0, 0., -75.0);
const static cv::Point3f P3D_MENTON(0., 0.,-133.0);

enum FACIAL_FEATURE {
    NOSE=30,
    RIGHT_EYE=36,
    LEFT_EYE=45,
    RIGHT_SIDE=0,
    LEFT_SIDE=16,
    EYEBROW_RIGHT=21,
    EYEBROW_LEFT=22,
    MOUTH_UP=51,
    MOUTH_DOWN=57,
    MOUTH_RIGHT=48,
    MOUTH_LEFT=54,
    SELLION=27,
    MOUTH_CENTER_TOP=62,
    MOUTH_CENTER_BOTTOM=66,
    MENTON=8
};

inline cv::Point2f toCv(const dlib::point& p)
{
    return cv::Point2f(p.x(), p.y());
}

HeadPose pose(full_object_detection &shape, float opticalPoseX, float opticalPoseY)
{
    int focalLength = 455;

    cv::Mat projectionMat = cv::Mat::zeros(3,3,CV_32F);
    cv::Matx33f projection = projectionMat;
    projection(0,0) = focalLength;
    projection(1,1) = focalLength;
    projection(0,2) = opticalPoseX;
    projection(1,2) = opticalPoseY;
    projection(2,2) = 1;

    std::vector<cv::Point3f> head_points;

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
    head_points.push_back(P3D_STOMMION);

    std::vector<cv::Point2f> detected_points;

    detected_points.push_back(toCv(shape.part(SELLION)));
    detected_points.push_back(toCv(shape.part(RIGHT_EYE)));
    detected_points.push_back(toCv(shape.part(LEFT_EYE)));
    detected_points.push_back(toCv(shape.part(RIGHT_SIDE)));
    detected_points.push_back(toCv(shape.part(LEFT_SIDE)));
    detected_points.push_back(toCv(shape.part(MENTON)));
    detected_points.push_back(toCv(shape.part(NOSE)));

    auto stomion = (toCv(shape.part(MOUTH_CENTER_TOP)) + toCv(shape.part(MOUTH_CENTER_BOTTOM))) * 0.5;
    detected_points.push_back(stomion);


    // Initializing the head pose 1m away, roughly facing the robot
    // This initialization is important as it prevents solvePnP to find the
    // mirror solution (head *behind* the camera)
    cv::Mat tvec = (cv::Mat_<double>(3,1) << 0., 0., 1000.);
    cv::Mat rvec = (cv::Mat_<double>(3,1) << 1.2, 1.2, -1.2);

    // Find the 3D pose of our head
    cv::solvePnP(head_points, detected_points,
            projection, cv::noArray(),
            rvec, tvec, true,
#ifdef OPENCV3
            cv::SOLVEPNP_ITERATIVE);
#else
            cv::ITERATIVE);
#endif

    cv::Matx33d rotation;
    Rodrigues(rvec, rotation);

    HeadPose headPose = {
        rotation(0,0),    rotation(0,1),    rotation(0,2),    tvec.at<double>(0)/1000,
        rotation(1,0),    rotation(1,1),    rotation(1,2),    tvec.at<double>(1)/1000,
        rotation(2,0),    rotation(2,1),    rotation(2,2),    tvec.at<double>(2)/1000,
                    0,                0,                0,                     1
    };

//#ifdef HeadPose_ESTIMATION_DEBUG


    std::vector<cv::Point2f> reprojected_points;

    projectPoints(head_points, rvec, tvec, projection, cv::noArray(), reprojected_points);

    //printf("rvec: [%f,%f,%f]\n",rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2));

    //for (auto point : reprojected_points) {
        //circle(_debug, point,2, Scalar(0,255,255),2);
    //}

    std::vector<cv::Point3f> axes;
    axes.push_back(cv::Point3f(0,0,0));
    axes.push_back(cv::Point3f(50,0,0));
    axes.push_back(cv::Point3f(0,50,0));
    axes.push_back(cv::Point3f(0,0,50));
    std::vector<cv::Point2f> projected_axes;

    projectPoints(axes, rvec, tvec, projection, cv::noArray(), projected_axes);

    //line(_debug, projected_axes[0], projected_axes[3], Scalar(255,0,0),2,CV_AA);
    //line(_debug, projected_axes[0], projected_axes[2], Scalar(0,255,0),2,CV_AA);
    //line(_debug, projected_axes[0], projected_axes[1], Scalar(0,0,255),2,CV_AA);

    //putText(_debug, "(" + to_string(int(pose(0,3) * 100)) + "cm, " + to_string(int(pose(1,3) * 100)) + "cm, " + to_string(int(pose(2,3) * 100)) + "cm)", coordsOf(face_idx, SELLION), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);

    printf("[%f,%f,%f]\n",(headPose(0,3) * 100),(headPose(1,3) * 100),(headPose(2,3) * 100));


//#endif

    return headPose;
}

int main()
{
    try
    {
        HeadPose headPose;

        Display *display;
        Window root;
        Screen* screen;
        double xSpeed=150,ySpeed=150;
        int dCenterMsk = 1;
        double poseX_,poseX,poseY_,poseY,dPoseX,dPoseY;
        int mouseX,mouseY;
        int win_x, win_y;
        unsigned int mask;
        Window ret_child;
        Window ret_root;

        poseX_=0,poseX=0;
        poseY_=0,poseY=0;
        //display = XOpenDisplay(0);
        //screen = DefaultScreenOfDisplay(display);
        //root = DefaultRootWindow(display);
        //XQueryPointer(display, root, &ret_root, &ret_child, &poseX_, &poseY_,
                //&win_x, &win_y, &mask);
        //XFlush(display);
        //XCloseDisplay(display);

        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            display = XOpenDisplay(0);
            screen = DefaultScreenOfDisplay(display);
            root = DefaultRootWindow(display);

            float opticalPoseX,opticalPoseY;
            // Grab a frame
            cv::Mat temp;
            cap >> temp;

            opticalPoseX = temp.cols / 2;
            opticalPoseY = temp.rows / 2;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
                shapes.push_back(pose_model(cimg, faces[i]));
                headPose = pose(shapes.at(i),opticalPoseX,opticalPoseY);

                //move pointer proportionally with face movement
                poseX = poseX_;
                poseX_ = headPose(0,3)*100;
                poseY = poseY_;
                poseY_ = headPose(1,3)*100;
                dPoseX = -(poseX_ - poseX);
                dPoseY = (poseY_ - poseY);
                root = XDefaultRootWindow(display);
                XQueryPointer(display, root, &ret_root, &ret_child, &mouseX, &mouseY,
                        &win_x, &win_y, &mask);
                mouseX = mouseX + dPoseX*xSpeed;
                mouseY = mouseY + dPoseY*ySpeed;
                XWarpPointer(display, None, root, 0, 0, 0, 0, mouseX, mouseY);
                XFlush(display);
            }


            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

