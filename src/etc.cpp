#include "yolo.h"
#include "cmdline.h"

static uchar calc_color(double temp1, double temp2, double temp3) {

    double color;
    if (6.0 * temp3 < 1.0)
        color = temp1 + (temp2 - temp1) * 6.0 * temp3;
    else if (2.0 * temp3 < 1.0)
        color = temp2;
    else if (3.0 * temp3 < 2.0)
        color = temp1 + (temp2 - temp1) * ((2.0 / 3.0) - temp3) * 6.0;
    else
        color = temp1;

    return (uchar)(color * 255);
}
void hsl2rgb(double hue, double sat, double light, uchar* rgb) {
    if (0.0 == sat) {
        rgb[0] = (uchar)(light * 255);
        rgb[1] = rgb[0];
        rgb[2] = rgb[0];
        return;
    }
    double temp1, temp2, temp3;
    if (light < 0.5) {
        temp2 = light * (1.0 + sat);
    }
    else {
        temp2 = light + sat - light * sat;
    }
    temp1 = 2.0 * light - temp2;
    temp3 = hue + 1.0 / 3.0;//for R, temp3=H+1.0/3.0	
    if (temp3 > 1.0)
        temp3 = temp3 - 1.0;
    rgb[0] = calc_color(temp1, temp2, temp3);
    temp3 = hue; //for G, temp3=H
    rgb[1] = calc_color(temp1, temp2, temp3);
    temp3 = hue - 1.0 / 3.0;//for B, temp3=H-1.0/3.0
    if (temp3 < 0.0)
        temp3 = temp3 + 1.0;
    rgb[2] = calc_color(temp1, temp2, temp3);

}
void draw_detections(Mat& frame, vector<DetectionObject>& objects) {
    uchar rgb[3];
    for (DetectionObject& obj : objects) {
        int xmin = (int)(obj.box.x - 0.5f * obj.box.w);
        int xmax = xmin + (int)obj.box.w;
        int ymin = (int)(obj.box.y - 0.5f * obj.box.h);
        int ymax = ymin + (int)obj.box.h;
        std::cout << "  -- Areca palm (confidence: " << fixed << setprecision(2) << obj.confidence * 100.0f <<
            "% @ (" << setw(3) << xmin << ", " << setw(3) << ymin << ", " << setw(3) << xmax << "," << setw(3) << ymax << "))\n";
       // ostringstream conf;
        hsl2rgb(obj.confidence, 0.8, 0.6, rgb);
        char buffer[100];
        sprintf_s(buffer, 100, "(%d)", obj.track_id);
        //conf << ":" << fixed << setprecision(2) << (obj.confidence * 100) << "%";
       putText(frame, buffer, Point(xmin, ymin - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
        rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(rgb[2], rgb[1], rgb[0]),2);
    } 
}
extern int info_x, info_y;
extern unsigned __int32 cap_text_color, cap_bg_color;
extern float cap_alpha;
void draw_titles(Mat& frame, float render, float wall, float detection, int frame_id ) {
    Mat bg = Mat::zeros(frame.size(), CV_8UC3);
    vector<string> strs;
    char buffer[200];
    //sprintf_s(buffer, 200, "Cap/Render time: %.2f ms", render);
   // strs.push_back(buffer);
    sprintf_s(buffer, 200, "Avg speed: %.2f ms (%.1f fps)", wall, 1000.0f / wall);
    strs.push_back(buffer);
    sprintf_s(buffer, 200, "Inference(%d) time Onboard: %.2f ms", frame_id, detection);
    strs.push_back(buffer);
    int bs = 0;   
    Point2f lt(info_x, info_y);
    unsigned char* p = reinterpret_cast<unsigned char*>(&cap_bg_color);
    Scalar bg_color(p[2], p[1], p[0]);
    for (int i = 0; i < strs.size(); i++) {
        Size sz = getTextSize(strs[i], FONT_HERSHEY_SIMPLEX, 0.75, 2, &bs);
        rectangle(bg, Rect(lt.x, lt.y - bs - 12, sz.width + 10, sz.height + 10), bg_color, FILLED); 
        lt.y += 30;
    }
    addWeighted(frame, 1, bg, cap_alpha, 0, frame);
    lt.y = info_y;
    p = reinterpret_cast<unsigned char*>(&cap_text_color);
    Scalar text_color(p[2], p[1], p[0]);
    for (int i = 0; i < strs.size(); i++) { 
        putText(frame, strs[i], lt, FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2);// Scalar(0x36, 0x41, 0xef), 2);
        lt.y += 30;
    }
}

void print_perf_counts(const map<string, InferenceEngineProfileInfo>& performanceMap,  const string& net_file) {
    long totalTime = 0;
    // Print performance counts

    XMLDocument doc;
    doc.LoadFile(net_file.c_str());
    XMLElement* layers = doc.RootElement()->FirstChildElement("layers/layer");
    ofstream of((net_file + ".perf.csv").c_str());
    of << "Layer, Type, InputDim, OutputDim, Filters, Groups, Execution,Time,ExeType\n";
    for (const auto& it : performanceMap) {

        if (it.second.realTime_uSec == 0)  continue;

        XMLElement* l = layers;
        while (l) {
            const char* n = l->Attribute("name");
            if (n && it.first == n) break; 
            l = l->NextSiblingElement();
        } 
        if (!l) {
            of << it.first << "," << it.second.layer_type  << " ,-, -, -, -,";
            
            switch (it.second.status) {
            case InferenceEngineProfileInfo::EXECUTED:
                of << "EXECUTED,";
                break;
            case InferenceEngineProfileInfo::NOT_RUN:
                of << "NOT_RUN,";
                break;
            case InferenceEngineProfileInfo::OPTIMIZED_OUT:
                of << "OPTIMIZED_OUT,";
                break;
            }
            of << it.second.realTime_uSec << ", " << it.second.exec_type << "\n"; 
            continue;
        }

        XMLElement* dim = l->FirstChildElement("input/port/dim");
        dim = dim->NextSiblingElement();
        int channels = dim->IntText();
         dim = dim->NextSiblingElement();
        int height = dim->IntText();
        dim = dim->NextSiblingElement();
        int width = dim->IntText();
        const char* layer_type = l->Attribute("type");
        of << it.first << "," << layer_type << "," << channels << "x" << height << "x" << width << ",";

        dim = l->FirstChildElement("output/port/dim");
        dim = dim->NextSiblingElement();
        channels = dim->IntText();
        dim = dim->NextSiblingElement();
        height = dim->IntText();
        dim = dim->NextSiblingElement();
        width = dim->IntText();
        of <<  channels << "x" << height << "x" << width << ",";
       
        if (layer_type && _strcmpi(layer_type, "Convolution") == 0) {
            XMLElement* data = l->FirstChildElement("data");
            int group = data->IntAttribute("group", 1); 
            of << "\"" << data->Attribute("output") << "(" << data->Attribute("kernel") << ")\"," << group << ",";
        }
        else {
            of << "-,- ,";
        } 
        switch (it.second.status) {
        case InferenceEngineProfileInfo::EXECUTED:
            of << "EXECUTED,";
            break;
        case InferenceEngineProfileInfo::NOT_RUN:
            of << "NOT_RUN,";
            break;
        case InferenceEngineProfileInfo::OPTIMIZED_OUT:
            of << "OPTIMIZED_OUT,";
            break;
        } 
        of << it.second.realTime_uSec << ", " << it.second.exec_type << "\n"; 
        
       totalTime += it.second.realTime_uSec;
       
    }
    of.close();
    cout<< fixed << setprecision(3)<< "\n *Total inference time on Myriad: "  << totalTime * 0.001f << " ms.\n\n";
}


ArgDef defs[] = {
    { "-ir", "network.xml", false, ir_message },
    { "-d", "MYRIAD", false, device_message },
    { "-pc", "", false, pc_message },
    { "-i", "", false, input_message },
    { "-t", "0.45", false, threshold_message},
    { "-nms", "0.5",false,nms_threshold_msg},
    { "-p", "", false, position_message},
    { "-c" ,"", false, color_message }
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
   // ofstream ifs("input.txt");
    //ifs << "FP32 [1, 3, 416, 416]\nbatch: 0\n" << fixed << setprecision(4);
    float* blob_data = blob->buffer().as<float*>(); 
    int index = 0;
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[index] = (float)temp.at<cv::Vec3b>(h, w)[c] / 255.0f;
               // ifs << blob_data[index] << " ";
                index++;
                
            }
          //  ifs << endl;
        }
        //ifs << endl;
    }
    //ifs.close();
}

double get_next_float(const char*& str) {
    const char* p = str;
    while (*p != ',' && *p != ' ' && *p != '\t' && *p != 0)
        p++;
    double r = atof(str);
    if (0 == *p)
        str = p;
    else {
        str = p + 1;
        while (*str == ',' || *str == ' ' || *str == '\t')
            str++;
    }
    return r;
}
int get_next_int(const char*& str) {
    const char* p = str;
    while (*p != ',' && *p != ' ' && *p != '\t' && *p != 0)
        p++;
    int r = atoi(str);
    if (0 == *p)
        str = p;
    else {
        str = p + 1;
        while (*str == ',' || *str == ' ' || *str == '\t')
            str++;
    }
    return r;
}
unsigned __int32 htoi(const char*& s) {
    if (*s == '#') {
        s++;
    }
    else if (*s == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }
    unsigned __int32 r = 0;
    while (*s) {
        if (*s >= '0' && *s <= '9') {
            r <<= 4;
            r += (*s - '0');
        }
        else if (*s >= 'a' && *s <= 'f') {
            r <<= 4;
            r += (*s - 'a' + 10);
        }
        else if (*s >= 'A' && *s <= 'F') {
            r <<= 4;
            r += (*s - 'A' + 10);
        }
        else if (*s == ',') {
            s++;
            break;
        }
        else
            break;
        s++;

    }
    return r;
}
