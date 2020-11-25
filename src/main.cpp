
#include "detect.h"
#include "yolo.h"
#include "cmdline.h"
#include "tinyxml2.h"
using namespace tinyxml2;
extern bool is_suffix(const char* filename, const char* ext);

extern void frame2blob(const Mat& image, Blob::Ptr& blob);
extern void dump_data(const string& layer_name, const Blob::Ptr& blob);
extern const char* replace_extension(string& str, const char* new_ext);
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

	ifstream f;
	XMLDocument doc;
	f.open(FLAGS_ir.param);
	if (!f) {
		cerr << " *** ir file open failed!\n";
		return -1;
	}
	f.close();
 
	f.open(FLAGS_i.param);
	if (!f) {
		cerr <<"Input file(a video or image) " << FLAGS_i.param << " does not exist!\n";
		return -1;
	} 
	f.close();
	


	try {
#ifdef _DEBUG
		Core core("plugins-debug.xml");
#else
		Core core("plugins.xml");
#endif
		vector<string> devices = core.GetAvailableDevices();
		bool found_MYRIAD = false;
		
		for (auto& d : devices) {
			if (d == "MYRIAD") {
				found_MYRIAD = true;
				break;
			}
		}
		if (!found_MYRIAD) {
			cerr << " *** MYRIAD VPU is not found. Plug and restart. *** \n\n";
			return -1;
		}
		map<string, string> configs = { { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } };
		core.SetConfig(configs, "MYRIAD");
		bool is_image = (is_suffix(FLAGS_i.param, ".jpg") || is_suffix(FLAGS_i.param, ".png")
			|| is_suffix(FLAGS_i.param, ".bmp") || is_suffix(FLAGS_i.param, ".gif"));

		cout << "\n Reading input..."  ;
		VideoCapture cap;
		Mat frame; 
		Size frame_size ;

		if (is_image) {
			frame = imread(FLAGS_i.param);
			frame_size = frame.size();
			cout << " ["<< frame_size.width <<" x " << frame_size.height <<"] image loaded.\n";
		}
		else {
			if (!((FLAGS_i.param == "cam") ? cap.open(0) : cap.open(FLAGS_i.param))) {
				cerr << " Cannot open input file or camera: " << FLAGS_i.param << endl;
				return -1;
			}
			cap >> frame;

			if (!cap.grab()) {
				cerr << "This demo supports only video (or camera) inputs !!! "
					"Failed to get next frame from the " << FLAGS_i.param << endl;
				return -1;
			}
			frame_size = frame.size();
			float fps = cap.get(CAP_PROP_FPS);
			cout << " [" << frame_size.width << " x " << frame_size.height << " fps "<< 
				fixed << setprecision(2)<<fps<<" ] video loaded.\n";
		}	

		string input_path(FLAGS_ir.param);

		cout << " Read Network and load to device ...";
		CNNNetwork network = core.ReadNetwork(input_path);
		ExecutableNetwork enetwork = core.LoadNetwork(network, "MYRIAD");
		cout << "Done\n";
		InputsDataMap input_info = network.getInputsInfo(); 

		if (input_info.size() != 1) {
			cerr << "\n *** Input layer number much be 1!\n\n"; 
			return -1;
		}
	

		InputInfo::Ptr& input_data = input_info.begin()->second; 
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);

		SizeVector dims = input_data->getTensorDesc().getDims();
		Size net_size = { (int)dims[2], (int)dims[3] };

 
		OutputsDataMap output_info = network.getOutputsInfo(); 
		for (auto& item : output_info) {  
			item.second->setPrecision(Precision::FP32);
		} 
		cout << "\n Parse yolo definition ...";
		YoloDetectionConfig yolo_config(replace_extension(input_path, ".yolo.xml")); 
		cout << " Done.\n Start inference ... \n";
		InferRequest infer_req = enetwork.CreateInferRequest();
		typedef chrono::duration<double, ratio<1, 1000>> ms;
		auto total_t0 = chrono::high_resolution_clock::now();
		auto wallclock = total_t0;
		double ocv_decode_time = 0, ocv_render_time = 0;

		const string& input_name = input_info.begin()->first;

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
				
				t0 = t1;
				ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
				wallclock = t0;
				vector<DetectionObject> objects;
				// Parsing outputs
				for (auto& output : output_info) {
					auto output_name = output.first;
					Blob::Ptr blob = infer_req.GetBlob(output_name);
					//dump_data(output_name, blob);
					parse_output(output_name, blob, objects, yolo_config, frame_size, net_size, threshold, nms_threshold);
				}
				// Drawing boxes
				draw_detections(frame, objects);

				ostringstream out;
				out << "OpenCV cap/render time: " << fixed << setprecision(2) << (ocv_decode_time + ocv_render_time) << " ms";
				putText(frame, out.str(), Point2f(0, 25), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0, 255, 0));
				out.str("");
				out << "Wallclock time "<< fixed << setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
				putText(frame, out.str(), Point2f(0, 50), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0, 0, 255));
				out.str("");
				out << "Detection time  : " << fixed << setprecision(2) << infer_time.count() << " ms ";
				putText(frame, out.str(), Point2f(0, 75), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(255, 0, 0));
			}
			imshow("Detection results", frame);

			t1 = chrono::high_resolution_clock::now();
			ocv_render_time = chrono::duration_cast<ms>(t1 - t0).count();

			const int key = waitKey(1);
			if (27 == key)  // Esc
				break;

			if (is_image) break;
			cap >> frame;
			if (frame.empty()) break;// end of video file
		}
		auto total_t1 = chrono::high_resolution_clock::now();
		ms total = chrono::duration_cast<ms>(total_t1 - total_t0);
		cout << "Total Inference time: " << total.count() << endl;
        /** Showing performace results **/
        
         print_perf_counts(infer_req.GetPerformanceCounts(), cout);
         
			
		
	}
	catch (const exception& e) {
		cerr << "[ ERROR ] " << e.what() << endl;
		return 1;
	}
	catch (...) {
		cerr << "[ ERROR ] Unknown/internal exception happened." << endl;
		return 1;
	}
	system("pause");
	return 0; 
}