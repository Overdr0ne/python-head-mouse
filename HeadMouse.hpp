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

//#define DEBUG

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

class HeadMouse {

public:
	HeadMouse();
	//std::vector<full_object_detection> update(cv_image<bgr_pixel> &cimg);
	std::vector<full_object_detection> update(cv::VideoCapture &cap, cv_image<bgr_pixel> &cimg);

private:
	Display *display;
	Window root;
	Screen *screen;
	int xSpeed,ySpeed;
	double poseX_,poseX,poseY_,poseY,dPoseX,dPoseY;
	int mouseX,mouseY;
	int centerX,centerY;
	int winX,winY;
	unsigned int mask;
	Window retChild,retRoot;
	int initX,initY;
	int xOffset,yOffset;
	float opticalPoseX,opticalPoseY;
	cv::Mat temp;
	cv::VideoCapture cap;
	int j;
	frontal_face_detector detector;
	shape_predictor pose_model;
	HeadPose headPose;

	HeadPose pose(full_object_detection &shape, float opticalPoseX, float opticalPoseY);
};

HeadMouse::HeadMouse() {
	xSpeed=20000,ySpeed=20000;
	xOffset=0,yOffset=0;
	poseX_=0,poseX=0;
	poseY_=0,poseY=0;
	display = XOpenDisplay(0);
	screen = DefaultScreenOfDisplay(display);
	root = DefaultRootWindow(display);
	centerX = WidthOfScreen(screen) / 2;
	centerY = HeightOfScreen(screen) / 2;
	initX = centerX;
	initY = centerY;
	j=0;

	// Load face detection and pose estimation models.
	detector = get_frontal_face_detector();
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
}

HeadPose HeadMouse::pose(full_object_detection &shape, float opticalPoseX, float opticalPoseY)
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

    headPose = {
        rotation(0,0),    rotation(0,1),    rotation(0,2),    tvec.at<double>(0)/1000,
        rotation(1,0),    rotation(1,1),    rotation(1,2),    tvec.at<double>(1)/1000,
        rotation(2,0),    rotation(2,1),    rotation(2,2),    tvec.at<double>(2)/1000,
                    0,                0,                0,                     1
    };

    std::vector<cv::Point2f> reprojected_points;

    projectPoints(head_points, rvec, tvec, projection, cv::noArray(), reprojected_points);

    std::vector<cv::Point3f> axes;
    axes.push_back(cv::Point3f(0,0,0));
    axes.push_back(cv::Point3f(50,0,0));
    axes.push_back(cv::Point3f(0,50,0));
    axes.push_back(cv::Point3f(0,0,50));
    std::vector<cv::Point2f> projected_axes;

    projectPoints(axes, rvec, tvec, projection, cv::noArray(), projected_axes);

#ifdef DEBUG
    printf("[%f,%f,%f]\n",(headPose(0,3) * 100),(headPose(1,3) * 100),(headPose(2,3) * 100));
#endif

    return headPose;
}

std::vector<full_object_detection>  HeadMouse::update(cv::VideoCapture &cap, cv_image<bgr_pixel> &cimg) {
	// Grab a frame
	cv::Mat temp;
	cap >> temp;

	opticalPoseX = temp.cols / 2;
	opticalPoseY = temp.rows / 2;

	cimg = cv_image<bgr_pixel>(temp);

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
		XQueryPointer(display, root, &retRoot, &retChild, &mouseX, &mouseY,
				&winX, &winY, &mask);
#ifdef DEBUG
		printf("centerx: %i\ncentery: %i\n",centerX,centerY);
#endif
		mouseX = mouseX + dPoseX*xSpeed;
		mouseY = mouseY + dPoseY*ySpeed;
		if(j==5) {
			xOffset = xSpeed*headPose(0,3);
			yOffset = ySpeed*headPose(1,3);
#ifdef DEBUG
			printf("xoffset: %i\n",xOffset);
			printf("yoffset: %i\n",yOffset);
#endif
		}
		mouseX = (centerX+xOffset) - xSpeed*headPose(0,3);
		mouseY = (centerY-yOffset) + ySpeed*headPose(1,3);
		XWarpPointer(display, None, root, 0, 0, 0, 0, mouseX, mouseY);
		XFlush(display);
		j++;
	}

	return shapes;
}
