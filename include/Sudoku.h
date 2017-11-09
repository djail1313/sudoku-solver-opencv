#include <cv.h>
#include <highgui.h>
#include "../knnclassification.h"

#ifndef SUDOKU_H
#define SUDOKU_H
#define SIZE 450

using namespace cv;
using namespace std;

class Sudoku
{
    public:
        Sudoku();
        virtual ~Sudoku();
        bool solve(Mat);

    protected:
        Mat preProcessImage(Mat);
        Mat crop(Mat);
        bool calculate(int,int);
        bool startSolve();
        bool checkValue(int,int,int);

    private:
        Mat image, croppedImage;
        DigitClassification *knn;
        int data[9][9];
};

#endif // SUDOKU_H
