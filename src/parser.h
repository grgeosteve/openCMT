#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

using std::string;
using std::vector;

struct Args {
    bool need_help;
    bool camera;
    int camera_code;
    bool pause;
    string video_filename;
    bool bbox_provided;
    cv::Rect bbox;
    bool estimate_scale;
    bool estimate_rotation;
    int skip;
};

class Parser {
public:
    Parser(int argc, char **argv);
    ~Parser();
    Args parse_args();
    void print_help();

private:
    Args args;
    int p_argc;
    char **p_argv;
    vector<string> p_args_list;

    bool pause_arg_used;
    bool video_filename_arg_used;
    bool bbox_arg_used;
    bool scale_arg_used;
    bool rotation_arg_used;
    bool skip_arg_used;
    bool camera_arg_used;
};

#endif
