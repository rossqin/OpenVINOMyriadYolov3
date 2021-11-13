#pragma once

const char ir_message[] = "Required. InferenceEngine Definition File(in xml)\n";
const char device_message[] = "Required. device 'CPU' or 'MYRAID'\n";
const char pc_message[] = "Per-layer metrics";
const char input_message[] = "Required. Path to input file.\n";
const char threshold_message[] = "Threshold, default 0.5\n";
const char nms_threshold_msg[] = "NMS threshold, default 0.8\n";
const char position_message[] = "Caption position, default is 10,25 ";
const char color_message[] = "Caption background & text colors";
struct ArgDef {
    const char* prefix;
    const char* param;
    bool exists;
    const char* hint;
};

void parse_cmd_line(int argc, char* argv[]);
extern ArgDef defs[];