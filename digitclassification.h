#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <fstream>

using namespace cv;
using namespace std;

class DigitClassification {

public:
    virtual bool trainCSV(char*) = 0;
    virtual bool trainImage(char*) = 0;
    virtual int classify(Mat) = 0;

protected:
    int num_images = 410, num_cols = 20, num_rows = 30;
    void preprocessImage(Mat*, Mat*, int);
    void removeBorder(Mat*);

};
