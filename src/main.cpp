#include "cmt.h"
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "parser.h"

using namespace cv;
using namespace std;

Point point1;
Point point2;
int drag = 0;
Rect rect;
Mat img;
int select_flag = 0;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        point1 = Point(x, y);
        drag = 1;
    }

    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        point2 = Point(x, y);
        Mat img1 = img.clone();
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("Select object", img1);
    }

    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect = Rect(point1.x, point1.y, x-point1.x, y-point1.y);
        drag = 0;
    }

    if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = 1;
        drag = 0;
    }
}

int main(int argc, char **argv)
{
    Parser parser(argc, argv);
    Args args = parser.parse_args();

    // Check if the help text is needed
    if (args.need_help) {
        parser.print_help();
        return -1;
    }

    int pause_time;
    if (args.pause) {
        pause_time = 0;
    }
    else {
        pause_time = 1;
    }

    VideoCapture cap;
    Mat frame;
    Mat frame_gray;
    int frame_count = 1;

    if (args.camera) {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "ERROR: Cannot initialize camera"
                 << endl;
            return -2;
        }
    }
    else {
        std::cout << args.video_filename << std::endl;
        cap.open(args.video_filename);
        if (!cap.isOpened()) {
            cerr << "ERROR: Cannot open video file"
                 << endl;
            return -2;
        }

        // Skip first frames if required
        if (args.skip > 0) {
            cap.set(CV_CAP_PROP_POS_FRAMES, args.skip);
        }
    }

    // Read first frame
    cap.read(frame);

    Rect bbox;
    if (args.bbox_provided) {
        bbox = args.bbox;
    }
    else if (args.camera) {
        int k;
        bool select = false;
        namedWindow("Camera feed", 1);
        while (1) {
            cap.read(frame);
            imshow("Camera feed", frame);
            if (select_flag == 1) {
                break;
            }
            k = waitKey(10);
            if (k == 'q')
            {
                select = false;
                break;
            }
            else if (k == 's')
            {
                select = true;
                break;
            }
        }
        destroyWindow("Camera feed");
        if (select) {
            img = frame.clone();
            namedWindow("Select object", 1);
            imshow("Select object", img);
            setMouseCallback("Select object", mouseHandler, NULL);
            cout << "Select the object bounding box from top-left to "
                 << "bottom-right and press any key..."
                 << std::endl;
            waitKey(0);
            bbox = rect;
            destroyWindow("Select object");
        }
        else {
            cerr << "Selection of object aborted" << endl;
            return -1;
        }
    }
    else {
        img = frame.clone();
        namedWindow("Select object", CV_WINDOW_FULLSCREEN);
        imshow("Select object", img);
        setMouseCallback("Select object", mouseHandler, NULL);
        cout << "Select the object bounding box from top-left to "
             << "bottom-right and press any key..."
             << std::endl;
        waitKey(0);
        bbox = rect;
        destroyWindow("Select object");
    }

    namedWindow("CMT Tracker", CV_WINDOW_FULLSCREEN);

    // Convert frame to grayscale
    cvtColor(img, frame_gray, COLOR_BGR2GRAY);

    // Initialize CMT
    CMT tracker = CMT(args.estimate_rotation,
                      args.estimate_scale,
                      "ORB",
                      "ORB",
                      512,
                      "BruteForce-Hamming",
                      20,
                      0.75,
                      0.8);

    // Print initial bounding box
    Point2f tl = bbox.tl();
    Point2f br = bbox.br();
    cout << "Using (" << tl.x << "," << tl.y
         << "," << br.x << "," << br.y << ")"
         << " as initial bounding box."
         << endl;

    // Initialize tracker
    int result;
    string errMessage;

    tracker.initialise(frame_gray, bbox, result, errMessage);
    while (1) {
        cap.read(frame);
        if (frame.empty()) {
            break;
        }
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Process the frame
        tracker.processFrame(frame_gray);

        // Display results
        Point2f center = tracker.getCenter();
        cout << "Frame " << frame_count
             << ":: center: " << std::setprecision(2) << std::fixed
             << "(" << center.x << "," << center.y << ")"
             << " | scale: " << tracker.getScaleEstimate()
             << " | rotation: " << tracker.getRotationEstimate()
             << " | active: " << tracker.getNumberOfActiveKeypoints()
             << endl;

        // Draw keypoints
        tracker.showAllInfo(frame);

        imshow("CMT Tracker", frame);

        char c = (char)waitKey(20);
        if (c == 'q') // Esc key
        {
            break;
        }
        ++frame_count;
    }
    destroyAllWindows();
    return 0;
}
