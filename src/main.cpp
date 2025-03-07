#include "inference.h"
#include "data.h"

int main(int argc, char **argv)
{
    TimeCounter time;
    cv::VideoCapture cap(2);

    if(!cap.isOpened()){
        cap.open(0);
        cerr << "Open camera deffault!" <<endl; 
    }

    // set ultralictics path 
    bool runOnGPU = false;
    int m_size = 320;
    string model_path = "/assets/best.onnx";

    // run inference yolo
    int WhiteBalanceValue = 4600;
    cv::namedWindow("WBT", cv::WINDOW_GUI_NORMAL);
    cv::createTrackbar("wb", "WBT", &WhiteBalanceValue, 6200);
    cap.set(cv::CAP_PROP_AUTO_WB, 0);   //turn off auto wb

    double currentWB = cap.get(cv::CAP_PROP_WB_TEMPERATURE);

    //load model yolo onnx
    Inference inf(model_path, cv::Size(m_size, m_size), "classes.txt", runOnGPU);

    int num_frames = 0;
    float posisi_line = 0.8;
    double obj_num = 0;
    clock_t start, end;
    double ms, fpsLive;

    while(true)
    {
        if(time.Count() >=1)
        {
            num_frames = 0;
            time.Reset();
        }
        num_frames++;
        start=clock();
        // set White Balance
        cap.set(cv::CAP_PROP_WB_TEMPERATURE, WhiteBalanceValue);
    
        cv::Mat frame;
        cap >> frame;

        int line_y = frame.rows * posisi_line;

        cv::Mat hls, mask;
        vector<Detection> output = inf.runInference(frame);
        
        int detections = output.size();

        for(int i=0 ; i<detections ; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            //bounder box
            cv::rectangle(frame, box, color, 2);
            
            string classString = detection.className + ' ' + to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            
            cv::rectangle(frame, textBox, color, cv::FILLED);   
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

            // COUNTER OBJECT
            int center_x = box.x + box.width / 2;
            int center_y = box.y + box.height / 2;
            int obj_id   = center_x + center_y * frame.cols;

            if (center_y > line_y && counts.find(obj_id) == counts.end())
            {
                counts.insert(obj_id);
                ++obj_num;
            }
        }

        //line batas 
        line(frame, cv::Point(0,0), cv::Point(100,100), cv::Scalar(255, 0, 0), 5);
        line(frame, cv::Point(0, line_y), cv::Point(frame.cols, line_y), cv::Scalar(0, 255, 0), 3);
        string counter_text = "Counter : " + to_string(obj_num);
        putText(frame, counter_text, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

        //inference end
        end= clock();

        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));

        double sc = (double(end) - double(start))/ double(CLOCKS_PER_SEC);
        fpsLive = double(num_frames) / double(sc);
        cout << "\t" << fpsLive << endl;

        cv::imshow("Inference", frame);
        if(cv::waitKey(1) == 27)
            break; 
    }

}