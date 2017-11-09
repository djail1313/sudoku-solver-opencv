#include "knnclassification.h"
#include <dirent.h>
#include <string>

using namespace std;
using namespace cv;

KNNClassification::KNNClassification(){
    knn = new KNearest();
}

KNNClassification::~KNNClassification(){
    delete knn;
}

bool KNNClassification::trainCSV(char *train_data){
    ifstream csv;
    int size = num_rows*num_cols, i = 0;
    Mat train_images = Mat(num_images, size, CV_32FC1);
    Mat train_classes = Mat(num_images, 1, CV_32FC1);
    string temp;

    csv.open(train_data);

    while(csv.good()){
        getline(csv, temp, ',');
        train_classes.at<float>(i, 0) = atof(temp.c_str());
        for(int x=0;x<size;x++){
            if(x < size-1) getline(csv, temp, ',');
            else getline(csv, temp, '\n');
            train_images.at<float>(i,x) = atof(temp.c_str());
        }
        i++;
    }
    csv.close();

    return knn->train(train_images, train_classes);
};

bool KNNClassification::trainImage(char *folder){
    int defaultSize = 50, x, y;
    string fname, lname;
    Mat img;
    int size = num_rows*num_cols, i = 0;
    Mat train_images = Mat(num_images, size, CV_32FC1);
    Mat train_classes = Mat(num_images, 1, CV_32FC1);
    DIR *pDIR;
    struct dirent *entry;

    if(pDIR = opendir(folder)){
        y = 0;
        while(entry = readdir(pDIR)){
            if(strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && y < num_images){
                lname = fname = entry->d_name;
                img = imread(string(folder)+"/"+fname, 0);
                if(!img.empty()){
                    fname = fname[0];
                    train_classes.at<float>(y, 0) = atof(fname.c_str());
                    resize(img, img, Size(defaultSize,defaultSize));
                    threshold(img, img, 128, 255, THRESH_BINARY);
                    if(countNonZero(img) > (img.cols*img.rows/2)){
                        bitwise_not(img, img);
                    }
                    preprocessImage(&img, &img, 1);
                    x = 0;
                    for(int i=0;i<img.rows;i++){
                        for(int j=0;j<img.cols;j++){
                            train_images.at<float>(y,x) = img.at<uchar>(i,j);
                            x++;
                        }
                    }
                    img.release();
                    y++;
                }
            }
        }
        closedir(pDIR);
    }
    return knn->train(train_images, train_classes);
};

int KNNClassification::classify(Mat img){
    threshold(img, img, 128, 255, THRESH_BINARY);
//    imwrite("training/training2/" + fname, img);
    preprocessImage(&img, &img, 0);
//    imwrite("training/training3/" + fname, img);
    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    erode(img, img, kernel);
    int count = countNonZero(img);
    if(count == 0){
        return 0;
    } else if(count > 400){
        erode(img, img, kernel);
    }
//    imwrite("training/training4/" + fname, img);
    img = img.reshape(1,1);
    return knn->find_nearest(Mat_<float>(img), 1);
}
