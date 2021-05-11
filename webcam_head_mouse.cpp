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

#include "HeadMouse.hpp"

using namespace dlib;
using namespace std;

int main() {
  try {
    cv::VideoCapture cap(0);
    HeadMouse headMouse;
    int j = 0;
    image_window win;
    cv_image<bgr_pixel> cimg;
    std::vector<full_object_detection> shapes;

    // Grab and process frames until the main window is closed by the user.
    while (!win.is_closed()) {
      shapes = headMouse.update(cap, cimg);

      // Display it all on the screen
      win.clear_overlay();
      win.set_image(cimg);
      win.add_overlay(render_face_detections(shapes));
    }
  } catch (serialization_error &e) {
    cout << "You need dlib's default face landmarking model file to run this "
            "example."
         << endl;
    cout << "You can get it from the following URL: " << endl;
    cout << "		"
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
         << endl;
    cout << endl << e.what() << endl;
  } catch (exception &e) {
    cout << e.what() << endl;
  }
}
