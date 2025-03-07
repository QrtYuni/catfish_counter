#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

set<int> counts;

struct TimeCounter{
    public:
        TimeCounter() : reset(true){}

        float Count()
        {
            if (reset)
            {
                previous_time = chrono::system_clock::now();
                reset = false;
            }
            current_time = chrono::system_clock::now();
            elapsed_time = current_time - previous_time;

            return elapsed_time.count();
        }

        void Reset(){
            reset = true;
        }

    private:
        chrono::time_point<chrono::system_clock> current_time, previous_time;
        chrono::duration<float> elapsed_time;

        bool reset;
};

#endif


