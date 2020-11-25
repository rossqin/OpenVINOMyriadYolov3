#pragma once 
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
    int  class_id;
    float confidence;

    DetectionObject(const Box& b, int cid, float conf);

    bool operator <(const DetectionObject& s2) const; 
    bool operator >(const DetectionObject& s2) const;
};
