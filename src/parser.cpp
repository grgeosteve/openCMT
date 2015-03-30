#include "parser.h"
#include <sstream>

Parser::Parser(int argc, char **argv)
{
    p_argc = argc;
    p_argv = argv;

    args.camera = true;
    args.estimate_rotation = true;
    args.estimate_scale = true;
    args.bbox_provided = false;
    args.need_help = false;
    args.skip = 0;

    video_filename_arg_used = false;
    bbox_arg_used = false;
    scale_arg_used = false;
    rotation_arg_used = false;
    pause_arg_used = false;
    camera_arg_used = false;
}

Parser::~Parser()
{

}

Args Parser::parse_args()
{
    if (p_argc == 1) {
        args.camera = true;
        args.pause = false;
        args.estimate_rotation = true;
        args.estimate_scale = true;
        args.bbox_provided = false;
        args.need_help = false;

        return args;
    }

    // Store arguments in a vector of strings
    p_args_list = vector<string>(p_argc - 1);
    for (int i = 1; i < p_argc; i++) {
        p_args_list[i - 1] = string(p_argv[i]);
    }

    for (int i = 0; i < p_args_list.size(); i++) {
        string arg = p_args_list[i];
        if (arg == "--help") {
            args.need_help = true;
            return args;
        }
        else if (arg == "--no-scale") {
            if (scale_arg_used) {
                std::cerr << "ERROR: Multiple --no-scale arguments"
                          << std::endl;
                args.need_help = true;
                return args;
            }
            scale_arg_used = true;
            args.estimate_scale = false;
        }
        else if (arg == "--without-rotation") {
            if (rotation_arg_used) {
               std::cerr << "ERROR:: Multiple --without-rotation arguments"
                         << std::endl;
                args.need_help = true;
                return args;
            }
            rotation_arg_used = true;
            args.estimate_rotation = false;
        }
        else if (arg == "--pause") {
            if (pause_arg_used) {
               std::cerr << "ERROR: Multiple --pause arguments"
                         << std::endl;
                args.need_help = true;
                return args;
            }
            if (bbox_arg_used) {
                std::cerr << "ERROR: Cannot use --pause and pass a bounding "
                          << "box with --bbox at the same time"
                          << std::endl
                          << "The object bounding box has already been provided "
                          << "with the --bbox"
                          << std::endl;
                args.need_help = true;
                return args;
            }
            pause_arg_used = true;
            args.pause = true;
       }
       else if (arg == "--camera") {
           if (video_filename_arg_used) {
               std::cerr << "ERROR: Cannot use --camera argument "
                         << "to enter custom camera code when a video file is used"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           if (camera_arg_used) {
               std::cerr << "ERROR: Multiple --camera arguments"
                         << std::endl;
                args.need_help = true;
                return args;
           }
           if ((i + 1) >= p_args_list.size()) {
               std::cerr << "ERROR: Incorrect use of --camera argument"
                         << std::endl
                         << "A following camera code is needed"
                         << std::endl
                         << "See help for more information"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           else {
               // Convert the next argument into an integer
               std::istringstream iss(p_args_list[i+1]);
               double d_cam_code;
               iss >> d_cam_code;

               // Check if the argument is numeric
               if (!iss) {
                   std::cerr << "ERROR: Invalid argument for --camera"
                             << std::endl
                             << "The camera code must be integer"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }
               // Check if argument is an integer
               else if (std::abs(floor(d_cam_code)) < std::abs(d_cam_code)) {
                   std::cerr << "ERROR: Invalid argument for --skip"
                             << std::endl
                             << "The camera code must be integer"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }
               else {
                   camera_arg_used = true;
                   args.camera = true;
                   args.camera_code = static_cast<int>(d_cam_code);
                   ++i;
               }
           }
       }
       else if (arg == "--filename") {
           if (video_filename_arg_used) {
               std::cerr << "ERROR: Multiple --filename arguments"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           if ((i + 1) >= p_args_list.size()) {
               std::cerr << "ERROR: Incorrect use of --filename argument"
                         << std::endl
                         << "A following filename is needed"
                         << std::endl
                         << "See help for more information"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           else {
               video_filename_arg_used = true;
               args.camera = false;
               args.video_filename = p_args_list[i + 1];
               ++i;
           }
       }
       else if (arg == "--bbox") {
           if (bbox_arg_used) {
                std::cerr << "ERROR: Multiple --bbox arguments"
                          << std::endl;
                args.need_help = true;
                return args;
            }
           if (pause_arg_used) {
                std::cerr << "ERROR: Cannot pass a bounding box with --bbox "
                          << "and pause the feed to select an extra bounding "
                          << "box at the same time"
                          << std::endl;
                args.need_help = true;
                return args;
            }
           if ((i + 1) >= p_args_list.size()) {
                std::cerr << "ERROR: Incorrect use of --bbox argument"
                          << "The argument must be followed by a continuous "
                          << "argument of comma-seperated coordinates of the "
                          << "initial bounding box"
                          << std::endl
                          << "See help for more information"
                          << std::endl;
                args.need_help = true;
                return args;
            }
           else {
               // Split the next argument into the coordinates of the
               // bounding box
               vector<string> coordinate_str_list;
               vector<float> coordinate_list;
               char delimitter = ',';
               string::size_type k = 0;
               string::size_type j = p_args_list[i+1].find(delimitter);

               int delimitter_count = 0;

               while (j != string::npos) {
                   ++delimitter_count;
                   if (delimitter_count > 3) { // More than 4 coords
                       std::cerr << "ERROR: Invalid argument for --bbox"
                                 << std::endl
                                 << "The number of comma-seperated coordinates "
                                 << "must be exactly four1"
                                 << std::endl;
                       args.need_help = true;
                       return args;
                   }
                   coordinate_str_list.push_back(p_args_list[i+1].substr(k, j));
                   k = ++j;
                   j = p_args_list[i+1].find(delimitter, j);

                   if (j == string::npos) {
                       coordinate_str_list.push_back(
                                   p_args_list[i+1].substr(k, p_args_list[i+1].length() - 1));
                   }
               }

               if (delimitter_count != 3) {
                   std::cerr << "ERROR: Invalid argument for --bbox"
                             << std::endl
                             << "The number of comma-seperated coordinates "
                             << "must be exactly four"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }

               coordinate_list = vector<float>(4);
               for (int n = 0; n < 4; n++) {
                   std::istringstream iss(coordinate_str_list[n]);
                   double d_coord;
                   iss >> d_coord;

                   // Check if the coordinate is numeric
                   if (!iss) {
                       std::cerr << "ERROR: Invalid argument for --bbox"
                                 << std::endl
                                 << "The coordinates of the bounding box "
                                 << "must be integers or floating-point numbers"
                                 << std::endl;
                       args.need_help = true;
                       return args;
                   }

                   coordinate_list[n] = static_cast<float>(d_coord);
               }

               // Check if the coordinates are valid bounding box
               // coordinates
               bool c1 = coordinate_list[0] < coordinate_list[2];
               bool c2 = coordinate_list[1] < coordinate_list[3];

               if (!(c1 && c2)) {
                   std::cerr << "ERROR: Invalid argument for --bbox"
                             << std::endl
                             << "The coordinates are not valid for a "
                             << "bounding box"
                             << std::endl
                             << "The coordinates passed must be:"
                             << std::endl
                             << "\tFirst the coordinates of the top-left bound"
                             << std::endl
                             << "\tSecond the coordinates of the bottom-right bound"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }
               // Passed all the tests; the coordinates are valid
               else {
                   bbox_arg_used = true;
                   args.bbox_provided = true;
                   cv::Point2f tl = cv::Point2f(coordinate_list[0],
                                                coordinate_list[1]);
                   cv::Point2f br = cv::Point2f(coordinate_list[2],
                                                coordinate_list[3]);
                   args.bbox = cv::Rect(tl, br);

                   ++i;
               }
           }
       }
       else if (arg == "--skip") {
           if (args.camera) { // no video filename has been specified
               std::cerr << "ERROR: Cannot use --skip argument "
                         << "to skip frames when camera feed is used"
                         << std::endl
                         << "A video filename must be specified "
                         << "with --filename before --skip argument"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           if (skip_arg_used) {
               std::cerr << "ERROR: Multiple --skip arguments"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           if ((i + 1) >= p_args_list.size()) {
               std::cerr << "ERROR: Incorrect use of --skip argument"
                         << std::endl
                         << "A following number of frames is needed"
                         << std::endl
                         << "See help for more information"
                         << std::endl;
               args.need_help = true;
               return args;
           }
           else {
               // Convert the next argument into an integer
               std::istringstream iss(p_args_list[i+1]);
               double d_frames;
               iss >> d_frames;

               // Check if the argument is numeric
               if (!iss) {
                   std::cerr << "ERROR: Invalid argument for --skip"
                             << std::endl
                             << "The number must be an integer"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }
               // Check if argument is positive
               else if (d_frames <= 0) {
                   std::cerr << "ERROR: Invalid argument for --skip"
                             << std::endl
                             << "The number must be a positive integer"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }
               // Check if argument is integer
               else if (floor(d_frames) < d_frames) {
                   std::cerr << "ERROR: Invalid argument for --skip"
                             << std::endl
                             << "The number must be a positive integer"
                             << std::endl;
                   args.need_help = true;
                   return args;
               }
               else {
                   skip_arg_used = true;
                   args.skip = static_cast<int>(d_frames);
                   ++i;
               }
           }
       }
       else {
           std::cerr << "ERROR: Cannot parse arguments"
                     << std::endl
                     << "See help for more information"
                     << std::endl;
           args.need_help = true;
           return args;
           break;
       }
    }

    return args;
}

void Parser::print_help()
{
    std::cout << std::endl
              << "CMT_Tracker" << std::endl
              << "Track an object using the CMT algorithm." << std::endl
              << "Usage: ./CMT_Tracker [OPTIONS]" << std::endl
              << "\tOPTIONS:" << std::endl;
    std::cout << "\t\t[--camera camCode]"
              << "\tUse a custom camera code instead of the default(0)"
              << std::endl;
    std::cout << "\t\t[--filename filename]"
              << "\ttrack an object in a video sequence"
              << " instead of a camera feed" << std::endl;
    std::cout << "\t\t[--no-scale]\tDon't estimate scale changes in the "
              << "object appearance"
              << " (The default behaviour is to estimate scale changes)"
              << std::endl;
    std::cout << "\t\t[--without-rotation]\tDon't estimate rotation in the "
              << "object appearance"
              << " (The default behaviour is to estimate rotation)"
              << std::endl;
    std::cout << "\t\t[--pause]\tPause the feed to select the bounding box "
              << "of the object to track" << std::endl;
    std::cout << "\t\t[--skip number_of_frames]"
              << "\tSkip the first n frames"
              << std::endl;
    std::cout << "\t\t[--bbox tl_x,tl_y,br_x,br_y]"
              << "\tSpecify initial bounding box"
              << std::endl;
    std::cout << "\t\t[--help]\tShow this text"
              << std::endl;
    std::cout << std::endl;

    std::cout << "Usage examples:"
              << std::endl;
    std::cout << "\t Use CMT_Tracker with camera feed with no other options."
              << std::endl;
    std::cout << "\t./CMT_Tracker"
              << std::endl;
    std::cout << "\t Use CMT_Tracker in a video sequence with a"
              << std::endl
              << "\t with a predefined bounding box"
              << std::endl;
    std::cout << "\t./CMT_Tracker --filename sample.avi --bbox 100,100,300,300"
              << std::endl;
    std::cout << "\t Use CMT_Tracker with camera feed with another camera(external webcam)"
              << "\t or if the camera in laptop that doesn't recognize the default(0) camera code"
              << "\t./CMT_Tracker --camera 100"
              << std::endl;

}
