 
#include "detect.h" 
#include <iostream>
#include <cmath>

using namespace std;

Box::Box(float center, float middle, float width, float height) {
	x = center;
	y = middle;
	w = width;
	h = height;
}
Box::Box(const float* var) {
	if (var) {
		x = var[0];
		y = var[1];
		w = var[2];
		h = var[3];
	}
	else
		x = y = w = h = 0.0;
}
 
static float overlap(float x1, float w1, float x2, float w2) {
	float l1 = x1 - w1 * 0.5;
	float l2 = x2 - w2 * 0.5;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 * 0.5;
	float r2 = x2 + w2 * 0.5;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}
float BoxIntersection(const Box& A, const Box& B) {
	float w = overlap(A.x, A.w, B.x, B.w);
	float h = overlap(A.y, A.h, B.y, B.h);
	if (w < 0 || h < 0) return 0.0;

	return w * h;
}
float BoxUnion(const Box& A, const Box& B) {
	float i = BoxIntersection(A, B);
	return A.Area() + B.Area() - i;
}
float BoxIoU(const Box& A, const Box& B) {
	float i = BoxIntersection(A, B);
	if (0.0 == i) return 0.0;
	float u = A.Area() + B.Area() - i;
	if (0.0 == u) {
		cerr << "Error: zero divisor in BoxIoU!\n";
		return 1.0;
	}
	return i / u;
}
float BoxIoUEx(const Box& A, const Box& B) {
	float i = BoxIntersection(A, B);
	if (0.0 == i) return 0.0;
	float size_b = B.Area();
	bool a_in_b = (size_b == i);
	float u = A.Area() + size_b - i;
	if (0.0 == u) {
		cerr << "Error: zero divisor in BoxIoU!\n";
		return 1.0;
	}
	return a_in_b ? (i / u) + 1.0f : (i / u);
}
float BoxRMSE(const Box& A, const Box& B) {
	float dx = A.x - B.x;
	float dy = A.y - B.y;
	float dw = A.w - B.w;
	float dh = A.h - B.h;

	return sqrt(dx * dx + dy * dy + dw * dw + dh * dh);

}


DetectionObject::DetectionObject(const Box& b, int cid, float conf) {
	box = b;
	class_id = cid;
	confidence = conf;
}

bool DetectionObject::operator <(const DetectionObject& s2) const {
	return this->confidence < s2.confidence;
}
bool DetectionObject::operator >(const DetectionObject& s2) const {
	return this->confidence > s2.confidence;
}

void calc_accuracy(const vector<DetectionObject>& objects,const vector<GTInfo>& gts,vector<PredictionInfo>& predictions) {
	PredictionInfo pdi;
	unsigned int prev_predicts = predictions.size();
	for (int d = 0; d < objects.size(); d++) {
		auto& det = objects.at(d);
		pdi.best_iou = 0.0f;
		pdi.gt_index = -1;
		pdi.class_id = -1;
		for (int g = 0; g < gts.size(); g++) {
			const GTInfo& gt = gts[g];
			Box gt_box(gt.x, gt.y, gt.w, gt.h);
			float iou = BoxIoU(gt_box, det.box);
			if (iou > pdi.best_iou) {
				pdi.best_iou = iou;
				pdi.gt_index = g;
			}
		}
		pdi.confidence = det.confidence;
		pdi.class_id = det.class_id;
		pdi.primary = (pdi.gt_index != -1 && det.class_id == gts[pdi.gt_index].class_id);
		if (pdi.primary) {
			for (unsigned int i = prev_predicts; i < predictions.size(); i++) {
				PredictionInfo& prev_p = predictions[i];
				if (prev_p.gt_index == pdi.gt_index) {
					// for one gt, we treat the prediction with the best_iou as primary prediction
					if (prev_p.best_iou >= pdi.best_iou) {
						pdi.primary = false;
					}
					else {
						prev_p.primary = false;
					}
				}
			}
		}
		predictions.push_back(pdi);
	}
}