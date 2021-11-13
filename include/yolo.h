#pragma once
#include <inference_engine.hpp>
#include <string>
#include <iostream>
#include <fstream>
#define HAVE_OPENCV_VIDEO
#include <opencv2/opencv.hpp>
#include "detect.h"
#include "tinyxml2.h"
using namespace tinyxml2;
using namespace InferenceEngine;
using namespace std;
using namespace cv;

#define MAX_YOLO_LAYER_BUFFER 32
struct YoloDetectionConfig;
struct YoloLayerConfig {
	char id[MAX_YOLO_LAYER_BUFFER];
	char base[MAX_YOLO_LAYER_BUFFER];
	int masks[MAX_YOLO_LAYER_BUFFER];
	YoloDetectionConfig* config;
};
struct YoloDetectionConfig {
	vector<string> classes;
	vector< pair<float, float> > anchors;
	vector <YoloLayerConfig> layers;
	YoloDetectionConfig(const char* filename);
}; 
int yolo_entry_index(int x, int y, int a, int c, int w, int h);
void parse_output(const string& layer_name, const Blob::Ptr& blob, vector<DetectionObject>& objects,
	const YoloDetectionConfig& yolo_config, const Size& img_sz, const Size& net_sz, float obj_threshold, float nms_threshold);

void draw_detections(Mat& frame, vector<DetectionObject>& objects);
void print_perf_counts(const map<string, InferenceEngineProfileInfo>& performanceMap, const string& filename);
void parse_yolo_definitions(const string& layer_name, XMLElement* root);
