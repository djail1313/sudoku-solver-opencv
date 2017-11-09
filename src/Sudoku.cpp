#include "Sudoku.h"

Sudoku::Sudoku()
{
    knn = new KNNClassification();
    bool r = knn->trainImage("training/images");
    if(!r){
        cout << "Train data gagal. Restart aplikasi." << endl;
    }
}

Sudoku::~Sudoku()
{
    delete knn;
}

bool Sudoku::solve(Mat img){
    image = img.clone();
    imshow("Original", image);

    img = preProcessImage(img);
    imshow("Preprocessing", img);

    img = crop(img);
    if(img.empty()) return false;

    Mat newimg;
    int digit;
    for(int m=0,i=0;m<SIZE,i<9;m+=50,i++){
        for(int n=0,j=0;n<SIZE,j<9;n+=50,j++){
            newimg = Mat(img, Rect(n,m,50,50));
            digit = knn->classify(newimg);
            newimg.release();
            data[i][j] = digit;
        }
    }
    bool result = calculate(0,0);
    for(int m=0,i=0;m<SIZE,i<9;m+=50,i++){
        for(int n=0,j=0;n<SIZE,j<9;n+=50,j++){
            digit = data[i][j];
            cout << digit << " ";
            ostringstream s;
            s << digit;
            putText(croppedImage, s.str(), Point(n+15,m+40), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(0,0,255),2);
        }
        cout << endl;
    }
    imshow("Result", croppedImage);
    return result;
};

Mat Sudoku::preProcessImage(Mat img){
    if(img.size().width > 800 && img.size().height > 600){
        resize(img, img, Size((img.size().width*20/100), img.size().height*20/100));
        image = img.clone();
    }
    cvtColor(img, img, CV_BGR2GRAY);
    GaussianBlur(img, img, Size(11,11), 0, 0);
    adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 2);
    return img;
};

Mat Sudoku::crop(Mat img){
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat mat, wrap;
    int area, maxarea, idx = -1;
    // Mendapatkan grid sudoku berdasarkan contour terbesar
    findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    maxarea = idx;
    // Mencari contour dengan area yang paling besar
    for(int i=0;i<contours.size();i++){
        area = contourArea(contours[i]);
        if(area > 50){
            if(area > maxarea){
                maxarea = area;
                idx = i;
            }
        }
    }

    // Menghitung perimeter -> panjang kurva
    double perimeter = arcLength(contours[idx], true);
    // Menggunakan algoritma Ramer–Douglas–Peucker algorithm
    // untuk membuang titik yang tidak diperlukan
    // hanya menyimpan titik pada setiap sudut
    // epsilon diisi dengan 1% dari parameter
    approxPolyDP(contours[idx], contours[idx], 0.01*perimeter, true);
    // contours[p] sekarang hanya mempunyai 4 titik
    // periksa jika tidak mendapatkan 4 titik maka gagal
    if(contours[idx].size() != 4){
        cout << contours[idx].size() << ". Gambar gagal diproses, karena kurang bagus." << endl;
        waitKey(0);
        Mat m;
        return m;
    }

    // #3 Croping image
    double sum = 0,
        diff1,
        diff2,
        diffprev = 0,
        diffprev2 = 0,
        prevsum = 0,
        prevsum2 = contours[idx][0].x + contours[idx][0].y;
    int a =0, b=0, c=0, d=0;
    // looping sebanyak sudut poligon persegi
    // Perlu menentukan titik titiknya karena titik di contours[p] index nya bisa berubah pada gambar yg lain
    for(int i=0;i<4;i++){
        sum = contours[idx][i].x + contours[idx][i].y;
        diff1 = contours[idx][i].x - contours[idx][i].y;
        diff2 = contours[idx][i].y - contours[idx][i].x;
        if(diff1 > diffprev){
            diffprev = diff1;
            c = i;
        }
        if(diff2 > diffprev2){
            diffprev2 = diff2;
            d = i;
        }
        if(sum > prevsum){
            prevsum = sum;
            a = i;
        }
        if(sum < prevsum2){
            prevsum2 = sum;
            b = i;
        }
    }
    // membuat point
    // in[4] digunakan untuk menentukan koordinat pada gambar sebelumnya
    Point2f in[4];
    // out[4] digunakan untuk menyiapkan gambar baru dengan ukuran tertentu
    Point2f out[4];
    in[0] = contours[idx][a];
    in[1] = contours[idx][b];
    in[2] = contours[idx][c];
    in[3] = contours[idx][d];
    out[0] = Point2f(SIZE,SIZE);
    out[1] = Point2f(0,0);
    out[2] = Point2f(SIZE,0);
    out[3] = Point2f(0,SIZE);
    // membuat gambar baru yang fokus di sudoku nya saja
    mat = Mat::zeros(mat.size(), mat.type());
    wrap = getPerspectiveTransform(in, out);
    warpPerspective(image, mat, wrap, Size(SIZE,SIZE));
    croppedImage = mat.clone();

    cvtColor(mat, mat, CV_BGR2GRAY);
    GaussianBlur(mat, mat, Size(11, 11), 0, 0);
    adaptiveThreshold(mat, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
    bitwise_not(img, img);
    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(img, img, kernel);
    erode(img,img,2);

    for(int i=0;i<450;i+=50){
        for(int j=0;j<450;j++){
            if(img.at<uchar>(i,j) != 0){
                floodFill(img, Point(i,j), Scalar(0));
            }
        }
    }

    for(int i=0;i<450;i++){
        for(int j=0;j<450;j+=50){
            if(img.at<uchar>(i,j) != 0){
                floodFill(img, Point(i,j), Scalar(0));
            }
        }
    }

    return img;
};

bool Sudoku::calculate(int index_i, int index_j){
    if(index_i > 8){
        return true;
    }
    if(index_j > 8){
        return calculate(index_i+1, 0);
    }
    if(data[index_i][index_j] != 0){
        return calculate(index_i, index_j+1);
    }
    int value = data[index_i][index_j];
    bool result = false;
    for(int i = 1; i <= 9; i++){
        if(checkValue(index_i, index_j, i)){
            data[index_i][index_j] = i;
            result = calculate(index_i, index_j+1);
        }
        if(result) return true;
    }
    if(!result) data[index_i][index_j] = value;
    return result;
};

bool Sudoku::checkValue(int index_i, int index_j, int value){
    // cek horizontal
    for(int j = 0; j < 9; j++){
        if(data[index_i][j] == value){
            return false;
        }
    }
    // cek vertical
    for(int i = 0; i < 9; i++){
        if(data[i][index_j] == value){
            return false;
        }
    }
    // cek kotak
    int i_min = 3 * (index_i/3);
    int j_min = 3 * (index_j/3);
    int i_max = i_min + 2;
    int j_max = j_min + 2;

    for(int i = i_min; i <= i_max; i++){
        for(int j = j_min; j <= j_max; j++){
            if(data[i][j] == value){
                return false;
            }
        }
    }
    return true;
};

