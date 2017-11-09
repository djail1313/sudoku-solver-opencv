#include "digitclassification.h"

using namespace cv;

void DigitClassification::removeBorder(Mat *img){
    int cols = img->cols;
    int rows = img->rows;
    for(int l=0;l<rows;l+=(rows-1)){
        for(int o=0;o<(cols/4);o++){
            floodFill(*img, Point(l,o), Scalar(0));
        }
        for(int o=(cols*3/4);o<(cols-1);o++){
            floodFill(*img, Point(l,o), Scalar(0));
        }
    }
    for(int l=0;l<rows;l+=(rows-1)){
        for(int o=0;o<(cols/4);o++){
            floodFill(*img, Point(o,l), Scalar(0));
        }
        for(int o=(cols*3/4);o<(cols-1);o++){
            floodFill(*img, Point(o,l), Scalar(0));
        }
    }
}

void DigitClassification::preprocessImage(Mat *inImage,Mat *outImage, int minarea = 0){
    Mat contourImage,regionOfInterest;

    vector<vector<Point> > contours;

    removeBorder(inImage);

    (*inImage).copyTo(contourImage);

    findContours(contourImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    int idx = -1;
    double area = 0, maxarea = -1;
    if(minarea == 0){
        minarea = inImage->cols*3;
    }
    for (int i = 0; i < contours.size(); i++){
        area = contourArea(contours[i]);
        if(area > minarea){
            if (maxarea < area)
            {
                idx = i;
                maxarea = area;
            }
        }
    }

    if(idx > -1){
        Rect rec = boundingRect(contours[idx]);
        regionOfInterest = (*inImage)(rec);
        resize(regionOfInterest,*outImage, Size(num_cols, num_rows));
    } else {
        *outImage = Mat::zeros(Size(num_cols, num_rows), 0);
    }
}
