#include<Windows.h>
#include<iostream>
#include <cstdlib>
#include <cstring>     // 用于 memset
using namespace std;
int n;
int a[1<<22];
int sum;

int main() {
    long long head, tail, freq;
    n = 1 << 22;



    srand(123);


    for (int i = 0; i < n; i++) {
        a[i] = rand() % 10000;
    }

    const int LOOP_TIMES = 1000;  // 循环1000次

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < LOOP_TIMES; k++) {

        sum = 0;

        /*
        for (int i = 0; i < n; i++) {
            sum += a[i];
        }*/
        
        int m = n;
        for (int t = 1; m > 1; t++) {
            int half = m / 2;
            for (int i = 0; i < half; i++) {
              a[i] = a[2 * i] + a[2 * i + 1];
            }
            if (m % 2 == 1) {
              a[half] = a[m - 1];
              m = half + 1;
            }
            else {
              m = half;
            }
        }
        
        
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    // 计算总时间 & 平均时间
    double total_ms = (tail - head) * 1000.0 / freq;
    double avg_ms = total_ms / LOOP_TIMES;

    cout << "总时间：" << total_ms << " ms" << endl;
    cout << "平均时间：" << avg_ms << " ms" << endl;

    return 0;
}