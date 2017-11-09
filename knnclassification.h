#include "digitclassification.h"

class KNNClassification: public DigitClassification {

public:
    KNNClassification();
    ~KNNClassification();
    bool trainCSV(char*);
    bool trainImage(char*);
    int classify(Mat img);

private:
    KNearest *knn;

};
