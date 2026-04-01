#include<Windows.h>
#include<iostream>
#include <cstdlib>
#include <cstring>     
using namespace std;

int n;
int res[5000];
int vector_[5000];    
int matrix[5000][5000];  
int main() {
    long long head, tail, freq;
    n = 5000;



    srand(123);

    for (int i = 0; i < n; i++) {
        vector_[i] = rand() % 10000;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10000;
        }
    }

    const int LOOP_TIMES = 100;  // 循环1000次

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);


    for (int k = 0; k < LOOP_TIMES; k++) {


        memset(res, 0, sizeof(res));


        
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                res[col] += matrix[row][col] * vector_[row];
            }
        }
        /*
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                res[col] += matrix[row][col] * vector_[row];
            }
        }*/
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);


    // 计算总时间 & 平均时间
    double total_ms = (tail - head) * 1000.0 / freq;
    double avg_ms = total_ms / LOOP_TIMES;

    cout << "总时间：" << total_ms << " ms" << endl;
    cout << "平均时间：" << avg_ms << " ms" << endl;

    return 0;
}