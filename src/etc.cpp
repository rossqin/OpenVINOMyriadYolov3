#include "yolo.h"
#include "cmdline.h"

void draw_detections(Mat& frame, vector<DetectionObject>& objects) {
    for (DetectionObject& obj : objects) {
        int xmin = (int)(obj.box.x - 0.5f * obj.box.w);
        int xmax = xmin + (int)obj.box.w;
        int ymin = (int)(obj.box.y - 0.5f * obj.box.h);
        int ymax = ymin + (int)obj.box.h;
        std::cout << "areca nut element with prob = " << obj.confidence <<
            "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")  will be rendered! \n";


        ostringstream conf;
        conf << ":" << fixed << setprecision(2) << (obj.confidence * 100) << "%";
        putText(frame, conf.str(), Point(xmin, ymin - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
        rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255));
    }

}


void print_perf_counts(const map<string, InferenceEngineProfileInfo>& performanceMap, ostream& stream, bool header) {
    long totalTime = 0;
    // Print performance counts
    if (header) {
        stream << "\nPerformance counts:\n\n";
    }
    for (const auto& it : performanceMap) {
        string to_print(it.first);
        const int max_layer_name = 30;

        if (it.first.length() >= max_layer_name) {
            to_print = it.first.substr(0, max_layer_name - 4);
            to_print += "...";
        }


        stream << setw(max_layer_name) << left << to_print;
        switch (it.second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            stream << setw(15) << left << "EXECUTED";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            stream << setw(15) << left << "NOT_RUN";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            stream << setw(15) << left << "OPTIMIZED_OUT";
            break;
        }
        stream << setw(30) << left << "LayerType: " + string(it.second.layer_type) + " ";
        stream << setw(20) << left << "RealTime: " + to_string(it.second.realTime_uSec);
        stream << setw(20) << left << " CPU: " + to_string(it.second.cpu_uSec);
        stream << " ExecType: " << it.second.exec_type << endl;
        if (it.second.realTime_uSec > 0) {
            totalTime += it.second.realTime_uSec;
        }
    }
    stream << setw(20) << left << "Total time: " + to_string(totalTime) << " clocks.\n";
}


ArgDef defs[] = {
    { "-ir", "network.xml", false, ir_message },
    { "-d", "MYRIAD", false, device_message },
    { "-pc", "", false, pc_message },
    { "-i", "", false, input_message },
    { "-t", "0.5", false, threshold_message},
    { "-nms", "0.8",false,nms_threshold_msg}
};

void parse_cmd_line(int argc, char* argv[]) {
    int arg_def_cnt = sizeof(defs) / sizeof(ArgDef);
    int i = 0;
    while (i < argc) {
        for (int j = 0; j < arg_def_cnt; j++) {
            if (_strcmpi(argv[i], defs[j].prefix) == 0) {
                defs[j].exists = true;
                if (++i < argc && argv[i][0] != '-') {
                    defs[j].param = argv[i];
                }
                break;
            }
        }
        i++;
    }
}

bool is_suffix(const char* filename, const char* ext) {
    size_t l1 = strlen(filename);
    size_t l2 = strlen(ext);
    if (l1 < l2) return false;
    const char* s = filename + (l1 - l2);
    return 0 == _strcmpi(s, ext);
}
string& trim(string& s) {
    if (s.empty())
        return s;
    s.erase(0, s.find_first_not_of(' '));
    s.erase(s.find_last_not_of(' ') + 1);
    return s;
}
void split_string(vector<string>& result, const string& str, char ch) {
    size_t off = 0;
    while (off < str.length()) {
        size_t pos = str.find(ch, off);
        if (pos == string::npos) pos = str.length();
        string t = str.substr(off, pos - off);
        trim(t);
        result.push_back(t);
        off = pos + 1;
    }
}

const char* replace_extension(string& str, const char* new_ext) {
    size_t pos = str.find_last_of('.');
    if (string::npos == pos) {
        str += new_ext;
        return str.c_str();
    }
    str.erase(pos);
    str += new_ext;
    return str.c_str();
}

void frame2blob(const Mat& image, Blob::Ptr& blob) {
    SizeVector blob_size = blob->getTensorDesc().getDims();
    const size_t width = blob_size[3];
    const size_t height = blob_size[2];
    const size_t channels = blob_size[1];
    Mat temp = Mat::zeros(Size(width, height), CV_8UC3);
    resize(image, temp, Size(width, height));

    uchar* blob_data = blob->buffer().as<uchar*>();
    uchar* img_data = temp.data;
    int index = 0;
    int image_size = height * width;
    for (int i = 0; i < image_size; i++) {
        for (int c = 0; c < channels; c++) {
            blob_data[c * image_size + i] = img_data[i * channels + c];
        }
    }

}