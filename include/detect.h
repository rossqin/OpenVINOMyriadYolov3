#pragma once 
#include <vector>
using namespace std;
enum {
    INDEX_PROBILITY_X = 0,
    INDEX_PROBILITY_Y,  // 1
    INDEX_PROBILITY_W,  // 2
    INDEX_PROBILITY_H,  // 3
    INDEX_CONFIDENCE,   // 4
    INDEX_PROBILITY_CLASS_0,
    INDEX_PROBILITY_CLASS_1,
    INDEX_PROBILITY_CLASS_2,
    INDEX_PROBILITY_CLASS_3
    //...
};
class Box {
public:
    float x;
    float y;
    float w;
    float h;
    Box(float center, float middle, float width, float height);
    Box(const float* var = nullptr);
    inline float Area() const { return w * h; }
};
float BoxIntersection(const Box& A, const Box& B);
float BoxUnion(const Box& A, const Box& B);
float BoxIoU(const Box& A, const Box& B);
float BoxRMSE(const Box& A, const Box& B);
float BoxIoUEx(const Box& A, const Box& B);

class DetectionObject {
public :
    Box box;
    int class_id;
    int track_id;
    float confidence;

    DetectionObject(const Box& b, int cid, float conf);

    bool operator <(const DetectionObject& s2) const; 
    bool operator >(const DetectionObject& s2) const;
};
struct GTInfo {
    int class_id;
    int x;
    int y;
    int w;
    int h;
};
struct PredictionInfo {
    float confidence;
    float best_iou;
    int class_id; 
    int gt_index;
    bool primary; // primary prediction of one gt 
};
inline bool compare_confidence(const PredictionInfo& p1, const PredictionInfo& p2) {
    return p1.confidence > p2.confidence;
}
void calc_accuracy(const vector<DetectionObject>& objects, const vector<GTInfo>& gts, vector<PredictionInfo>& predictions);
