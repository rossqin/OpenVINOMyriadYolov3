 
#include "detect.h"
#include "yolo.h"
#include "cmdline.h"
#include "tinyxml2.h" 
#include <direct.h>
#include <io.h>
#include <fstream>
#include <vector>
#include "tracking.h"

#define SPLIT_CHAR '\\'

using namespace tinyxml2;
extern bool is_suffix(const char* filename, const char* ext);

extern void frame2blob(const Mat& image, Blob::Ptr& blob);
extern void dump_data(const string& layer_name, const Blob::Ptr& blob);
extern const char* replace_extension(string& str, const char* new_ext);
extern void draw_titles(Mat& frame, float render, float wall, float detection, int frame_id);
int info_x = 10, info_y = 25; 
unsigned __int32 cap_text_color = 0xDAA520, cap_bg_color = 0x00FF7F;
float cap_alpha = 0.75f;

extern unsigned __int32 htoi(const char*& s);
extern double get_next_float(const char*& str);
extern int get_next_int(const char*& str);

struct SavedBox {
	float x, y, w, h;
	int track_id;
};

int main(int argc, char* argv[]) {
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif	 

	const char* command = argv[1];
	argc--;
	argv++;
	parse_cmd_line(argc, argv);
	ArgDef& FLAGS_ir = defs[0] ;
	ArgDef& FLAGS_device = defs[1];
	ArgDef& FLAGS_pc = defs[2];
	ArgDef& FLAGS_i = defs[3];
     
    float threshold = atof(defs[4].param);
    float nms_threshold = atof(defs[5].param);

	if (defs[6].exists) {
		const char* s = strchr(defs[6].param, ',');
		if (s) {
			info_x = atoi(defs[6].param);
			info_y = atoi(s + 1);
		}
	}
	if (defs[7].exists) {
		const char* s = defs[7].param; 
		if (*s) {
			cap_text_color = htoi(s);
			cap_bg_color = htoi(s);
			if (*s) cap_alpha = atof(s);
		}
	}
	vector<GTInfo> gts;
	vector<PredictionInfo> predictions;
	ifstream f;
	XMLDocument doc;

	vector<string> files;
	double all_wc_time = 0;
	// FLAGS_i is a folder or a file?
	struct stat s = { 0 };
	string folder(FLAGS_i.param);
	stat(FLAGS_i.param, &s);
	if (s.st_mode & _S_IFDIR) {
		char c = FLAGS_i.param[folder.length() - 1];
		if (c != '/' && c != '\\')
			folder += SPLIT_CHAR;
		string search_str = folder + "*.*";
		_finddata_t find_data;
		intptr_t handle = _findfirst(search_str.c_str(), &find_data);
		if (handle == -1) {
			cerr << "Error: Failed to find first file under `" << folder.c_str() << "`!\n";
			return false;
		}
		bool cont = true;

		while (cont) {
			if (0 == (find_data.attrib & _A_SUBDIR)) {
				if (is_suffix(find_data.name, ".jpg") ||
					is_suffix(find_data.name, ".JPG") ||
					is_suffix(find_data.name, ".png") ||
					is_suffix(find_data.name, ".PNG") ||
					is_suffix(find_data.name, ".bmp") ||
					is_suffix(find_data.name, ".BMP")
					) {
					files.push_back(folder + find_data.name);
				}
			}
			cont = (_findnext(handle, &find_data) == 0);
		}
		_findclose(handle);

	}
	else
		files.push_back(FLAGS_i.param);


 	f.open(files[0].c_str());
	if (!f) {
		cerr <<"Input file(a video or image) " << FLAGS_i.param << " does not exist!\n";
		return -1;
	} 
	f.close();

	try {
		char path[_MAX_PATH];
		strcpy_s(path,_MAX_PATH, _pgmptr);
		//GetModuleFileName(NULL, path, _MAX_PATH);
		char* p_str = path;
		for (int i = 0; path[i] != 0; i++) {
			if (path[i] == '/' || path[i] == '\\') {
				p_str = path + (i + 1);
			}
		}
#ifdef _DEBUG
		strcpy(p_str, "plugins-debug.xml");
		
#else
		strcpy(p_str, "plugins.xml");
#endif
		Core core(path);
		cout << "\n Looking for MYRIAD...";
		vector<string> devices = core.GetAvailableDevices();
		bool found_MYRIAD = false;
		
		for (auto& d : devices) {
			if (d == "MYRIAD") {
				found_MYRIAD = true;
				cout << " Found.\n";
				break;
			}
		}
		if (!found_MYRIAD) {
			cerr << " *** MYRIAD VPU is not found. Plug and restart. *** \n\n";
			return -1;
		}
		map<string, string> configs = { { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } };
		core.SetConfig(configs, "MYRIAD");
		
		string input_path(FLAGS_ir.param);

		cout << " Read IR model and transferred to device ...\n";
		CNNNetwork network = core.ReadNetwork(input_path);

		ExecutableNetwork enetwork = core.LoadNetwork(network, "MYRIAD");
		cout << " Model parameters transferred.\n";
		InputsDataMap input_info = network.getInputsInfo();

		string yolo_path(input_path);
		YoloDetectionConfig yolo_config(replace_extension(yolo_path, ".yolo.xml"));
		if (input_info.size() != 1) {
			cerr << "\n *** Input layer number much be 1!\n\n";
			return -1;
		}

		InputInfo::Ptr& input_data = input_info.begin()->second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);

		SizeVector dims = input_data->getTensorDesc().getDims();
		Size net_size = { (int)dims[2], (int)dims[3] };

		OutputsDataMap output_info = network.getOutputsInfo();
		for (auto& item : output_info) {
			item.second->setPrecision(Precision::FP32);
		}

		cout << "\n Reading input..."  ;
		VideoCapture cap;
		Mat frame; 
		Size frame_size ;
		float tp = 0.0f, fp = 0.0f, avr_iou = 0.0f;
		int all_gts = 0;
		InferRequest infer_req = enetwork.CreateInferRequest();
		vector< TrackingTraget> targets;
		int current_frame_id = 0; 
		auto start_clock = chrono::high_resolution_clock::now();
		int total_frame_so_far = 1;
		for (int i = 0; i < files.size(); i++) {
			string file = files[i] ;
			bool is_image = (is_suffix(file.c_str(), ".jpg") || is_suffix(file.c_str(), ".png") || is_suffix(file.c_str(), ".bmp") || is_suffix(file.c_str(), ".gif"));

			if (is_image) {
				frame = imread(file);
				frame_size = frame.size();
				total_frame_so_far++;
				cout << " [" << frame_size.width << " x " << frame_size.height << "] image loaded.\n";
				if (frame_size.width > 1200 || frame_size.height > 800) {
					float r_w = frame_size.width / 1200.0f;//4.56
					float r_h = frame_size.height / 800.0f;
					if (r_w > r_h) {
						frame_size.width = 1200;
						frame_size.height = (int)((float)frame_size.height / r_w);
					}
					else {
						frame_size.height = 800;
						frame_size.width = (int)((float)frame_size.width / r_h);
					}
					Mat temp = frame;
					frame = Mat::zeros(frame_size, CV_8UC3);
					resize(temp, frame, frame_size);
				} 
				const char* txt = replace_extension(file, ".txt");
				ifstream f(txt);
				if (f.is_open()) {
					char line[512];
					GTInfo info;
					while (!f.eof()) {
						f.getline(line, 512);
						const char* line_str = line;
						info.class_id = get_next_int(line_str);
						info.x = (int)(frame_size.width * get_next_float(line_str));
						info.y = (int)(frame_size.height * get_next_float(line_str));
						info.w = (int)(frame_size.width * get_next_float(line_str));
						info.h = (int)(frame_size.height * get_next_float(line_str));
						gts.push_back(info);
					}
					f.close();
				}
			}
			else {
				if (!((file == "cam") ? cap.open(0) : cap.open(file))) {
					cerr << " Cannot open input file or camera: " << file << endl;
					return -1;
				}
				cap >> frame;

				if (!cap.grab()) {
					cerr << "This demo supports only video (or camera) inputs !!! "
						"Failed to get next frame from the " << file << endl;
					return -1;
				}
				frame_size = frame.size();
				float fps = cap.get(CAP_PROP_FPS);
				cout << " [" << frame_size.width << " x " << frame_size.height << " fps " <<
					fixed << setprecision(2) << fps << " ] video loaded.\n";
			}

			

			cout << "\n Download image "<<i<<" data & start inference ... \n";
			ofstream ofs("detections.dat",ios_base::binary);
			typedef chrono::duration<double, ratio<1, 1000>> ms;
			
			auto wallclock = chrono::high_resolution_clock::now();
			
			double ocv_decode_time = 0, ocv_render_time = 0;

			const string& input_name = input_info.begin()->first;
			SavedBox sb = { 0,0,0,0,-1 };
			while (true) {
				auto t0 = chrono::high_resolution_clock::now();
				Blob::Ptr blob = infer_req.GetBlob(input_name);
				frame2blob(frame, blob);

				auto t1 = chrono::high_resolution_clock::now();
				ocv_decode_time = chrono::duration_cast<ms>(t1 - t0).count();
				t0 = chrono::high_resolution_clock::now();
				infer_req.StartAsync();

				if (OK == infer_req.Wait(IInferRequest::WaitMode::RESULT_READY)) {
					t1 = chrono::high_resolution_clock::now();
					ms infer_time = chrono::duration_cast<ms>(t1 - t0);
					cout << " Result ready in " << infer_time.count() << " ms.\n\n";
					t0 = t1;
					ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
					wallclock = t0;
					all_wc_time += wall.count();
					vector<DetectionObject> objects;
					// Parsing outputs
					for (auto& output : output_info) {
						auto output_name = output.first;
						Blob::Ptr blob = infer_req.GetBlob(output_name);
						parse_output(output_name, blob, objects, yolo_config, frame_size, net_size, threshold, nms_threshold);
					}
					if (is_image) {
						calc_accuracy(objects, gts, predictions);
					}
					//TrackDetection(targets, objects, current_frame_id);
					//ofs << current_frame_id << ":";
					
					for (auto o : objects) {
						sb.x = o.box.x;
						sb.y = o.box.y;
						sb.w = o.box.w;
						sb.h = o.box.h; 
						ofs.write(reinterpret_cast<char*>(&sb), sizeof(sb));
					}
					sb.x = sb.y = sb.h = sb.w = 0;
					ofs.write(reinterpret_cast<char*>(&sb), sizeof(sb));
					// Drawing boxes
					draw_detections(frame, objects);
					current_frame_id++; 
					float tm = 0;
					for (auto it : infer_req.GetPerformanceCounts()) {
						tm += it.second.realTime_uSec;
					}
					auto draw_t = chrono::high_resolution_clock::now();
					double time_so_far = chrono::duration_cast<ms>(draw_t - start_clock).count() / total_frame_so_far;
					draw_titles(frame, ocv_decode_time + ocv_render_time, time_so_far, tm * 0.001f, current_frame_id);


				}
				imshow("Detection Results", frame);
				t1 = chrono::high_resolution_clock::now();
				ocv_render_time = chrono::duration_cast<ms>(t1 - t0).count();

				const int key = waitKey(100);
				if (27 == key)  // Esc
					break;

				if (is_image) break;
				cap >> frame;
				total_frame_so_far++;
				if (frame.empty()) break;// end of video file
				imshow("Detection Results", frame);
				cap >> frame;
				if (frame.empty()) break;// end of video file
				total_frame_so_far++;
				if (total_frame_so_far > 600) break;
			}
			
			ofs.close();
			
			if (gts.size() > 0) { 
				for (auto& pdi : predictions) {
					//if (pdi.class_id != c) continue;
					if (pdi.primary && pdi.best_iou >= 0.5f) {
						tp += 1.0;
						avr_iou += pdi.best_iou;
					}
					else
						fp += 1.0f;
				}
				all_gts += gts.size();
				
				
			}
			
		}
		if (all_gts > 0) {
			float r = tp / all_gts;
			float p = tp / (tp + fp);
			float f1 = (2 * r * p) / (r + p);
			avr_iou /= all_gts;
			double fps = 1000.0 * files.size() / all_wc_time;
			cout << "\n *** R : " << setprecision(2) << r * 100.0f << "%, P: " << p * 100.0f << "%, F1: " << setprecision(4) << f1 << ". IoU: " << avr_iou << ", " << setprecision(1) << fps << " fps.\n";
			/** Showing performace results **/
			//print_perf_counts(infer_req.GetPerformanceCounts(), input_path);
		}
	}
	catch (const exception& e) {
		cerr << "\n\n[ ERROR ] " << e.what() << endl;
		return 1;
	}
	catch (...) {
		cerr << "\n\n[ ERROR ] Unknown/internal exception happened." << endl;
		return 1;
	}
	//system("pause");
	return 0; 
}