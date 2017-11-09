#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <fstream>
#include "Sudoku.h"
#define SIZE 450

using namespace std;
using namespace cv;

int main()
{
    Sudoku *sudoku = new Sudoku();
    Mat img;
    string test;
    while(cout<<"Masukkan path gambar (tidak boleh ada spasi) : ", cin>>test, test!="exit"){
        img = imread(test, 1);
        if(!img.empty()){
            if(sudoku->solve(img)){
                cout << "Solusi dapat ditemukan." << endl;
            } else {
                cout << "Solusi tidak dapat ditemukan." << endl;
            }
            waitKey(0);
        } else {
            cout << "Gambar tidak dapat ditemukan." << endl;
        }
    }

    return 0;
}
