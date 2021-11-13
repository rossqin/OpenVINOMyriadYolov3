
#include "detect.h"
#include "yolo.h"
#include <iomanip>

int yolo_entry_index(int classes, int x, int y, int a, int c, int w, int h) {
    int channels =  (classes == 1) ? 5 : (5 + classes);
    return (a * channels + c) * (w * h) + y * w + x;
}
float sigmoid(float x) {
    return 1.0f / (1.0f + exp( -x));
}
void dump_data(const string& layer_name, const Blob::Ptr& blob) {
    SizeVector dims = blob->getTensorDesc().getDims();
    float* output = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    string fname = layer_name + ".data.txt";
    ofstream ofile(fname);
    ofile << "FP32 [" << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << "]\n";
    ofile << "batch: 0\n";
    ofile << fixed << std::setprecision(4);
    int index = 0;
    for (int c = 0; c < dims[1]; c++) {
        for (int h = 0; h < dims[2]; h++) {
            for (int w = 0; w < dims[3]; w++) {
                ofile <<  output[index++] << ' ';
            }
            ofile << endl;
        }
        ofile << endl;
    }
    ofile.close();
}
void parse_output(const string& layer_name, const Blob::Ptr& blob, vector<DetectionObject>& objects,
    const YoloDetectionConfig& yolo_config, const Size& img_sz, const Size& net_sz,
    float obj_threshold, float nms_threshold) { 
    vector< pair<float, float> > anchors;
    for (auto& l : yolo_config.layers) {
        if (layer_name == l.base) {
            int i = 0; 
            while (l.masks[i] >= 0) {
                anchors.push_back(yolo_config.anchors[l.masks[i]]);
                i++;
            }
            break;
        }
    }

    SizeVector dim = blob->getTensorDesc().getDims();
    int out_blob_h = dim[2];
    int out_blob_w = dim[3];

    float expand_w = (float)img_sz.width / (float)net_sz.width;
    float expand_h = (float)img_sz.height / (float)net_sz.height;
     
    float* output = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

    int classes = yolo_config.classes.size();
    
    //TODO: 
    int cells = out_blob_h * out_blob_w;
    for (int y = 0; y < out_blob_h; y++) {
        for (int x = 0; x < out_blob_w; x++) {
            for (int a = 0; a < anchors.size(); a++) {
                int x_index = yolo_entry_index(classes , x, y, a, INDEX_PROBILITY_X, out_blob_w, out_blob_h);
                int conf_index = x_index + (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * cells;
                float confidence = sigmoid(output[conf_index]);
                if (confidence < obj_threshold) continue;
                int class_id = (classes == 1) ? 0 : -1;
                if (classes > 1) {
                    float best_cls_conf = 0.0f;
                    for (int c = 0, c_index = conf_index + cells; c < classes; c++, c_index += cells) {
                        float c_conf = sigmoid(output[c_index]);
                        if (c_conf  > best_cls_conf) {
                            class_id = c;
                            best_cls_conf = c_conf;
                        }
                    }
                }
                float cx = 1.0f / (1.0f + exp(-output[x_index])) ;
                float cy = 1.0f / (1.0f + exp(-output[x_index + cells]));
                float cw = exp(output[x_index + 2 * cells]);
                float ch = exp(output[x_index + 3 * cells]);
                Box pred_box((x + cx) / out_blob_w * img_sz.width, 
                    (y + cy) / out_blob_h * img_sz.height,
                    anchors[a].first * expand_w * cw,
                    anchors[a].second * expand_h * ch);
                bool to_add = true;
                for (DetectionObject& obj : objects) {
                    if (class_id != obj.class_id) continue;
                    float iou = BoxIoU(pred_box, obj.box);
                    if (iou > nms_threshold) { //merge
                        to_add = false;
                        if (confidence > obj.confidence) {
                            //replace 
                            obj.confidence = confidence;
                            obj.box = pred_box;
                        }
                        //otherwise forget about this new prediction
                    }
                }
                if (to_add) {
                    DetectionObject new_obj(pred_box, class_id, confidence);
                    objects.push_back(new_obj);
                }


            }
        }
    }

}
#if 0
extern void split_string(vector<string>& result, const string& str, char ch);
void parse_yolo_definitions(const string& layer_name, map<string, YoloLayerData>& results, XMLElement* root) {
    XMLElement* e = root->LastChildElement();
    while (e) {
        if (layer_name == e->Attribute("name")) {
            e = e->FirstChildElement("data");
            if (e) {
                const char* a_str = "";
                const char* m_str = "";
                YoloLayerData data;

                e->QueryIntAttribute("classes", &data.classes);
                e->QueryStringAttribute("anchors", &a_str);
                e->QueryStringAttribute("mask", &m_str);
                vector<string> strs ;
                split_string(strs, string(a_str), ',');
                if (strs.size() % 2) {
                    cerr << "Wrong anchors!\n";
                    break;
                }
                data.anchors.clear();
                vector< pair<float, float> > temp;
                for (int i = 0; i < strs.size(); i += 2) {
                    temp.push_back(make_pair(atof(strs[i].c_str()), atof(strs[i + 1].c_str())));
                }
                strs.clear();
                split_string(strs, string(m_str), ',');
                for (int i = 0 ; i < temp.size() ; i++) {
                    for (int j = 0; j < strs.size(); j++) {
                        if (atoi(strs[j].c_str()) == i) {
                            data.anchors.push_back(temp[i]);
                            break;
                        }
                    }
                }
                results.insert(make_pair(layer_name, data));

            }
            break;
        }
        e = e->PreviousSiblingElement();
    }
}
#endif
extern void split_string(vector<string>& result, const string& str, char ch);
YoloDetectionConfig::YoloDetectionConfig(const char* filename) {
    XMLDocument doc;
    if (XML_SUCCESS != doc.LoadFile(filename)) {
        throw exception("Failed to load yolo configures!");
    }
    XMLElement* root = doc.RootElement();
    XMLElement* classElement = root->FirstChildElement("classes/class");
    while (classElement) {
        const char* class_name = classElement->GetText();
        if (!class_name) class_name = "<NONAME>";
        classes.push_back(class_name);
        classElement = classElement->NextSiblingElement();
    }
    if (classes.size() == 0) {
        throw exception(" Error: class count should > 0  in a yolo config file !"); 
    }
    XMLElement* anchorElement = root->FirstChildElement("anchors/anchor");       
    while (anchorElement) {
        float w = anchorElement->FloatAttribute("width", 0.0f);
        float h = anchorElement->FloatAttribute("height", 0.0f);
        if (w > 0.0 && h > 0.0) {
            anchors.push_back(pair<float, float>(w, h));
        }
        anchorElement = anchorElement->NextSiblingElement();
    }
    XMLElement* outputElement = root->FirstChildElement("outputs/output");
    YoloLayerConfig c = { "","",{-1}, this };
    while (outputElement) {
        const char* id = outputElement->Attribute("id");
        strcpy_s(c.id, MAX_YOLO_LAYER_BUFFER, id);
        const char* base = outputElement->Attribute("base");
        strcpy_s(c.base, MAX_YOLO_LAYER_BUFFER, base);
        string masks_s = outputElement->Attribute("anchor-masks");
        vector<string> mask_strs;
        split_string(mask_strs, masks_s, ',');        
        int i = 0;        
        for (string& m : mask_strs) {
            c.masks[i++] = atoi(m.c_str());
        }
        c.masks[i++] = -1;
        layers.push_back(c);
        outputElement = outputElement->NextSiblingElement();
    }
     
}
