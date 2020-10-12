#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include<algorithm>

void TorchTest(){
    torch::jit::script::Module module = torch::jit::load("../model.pt");
    // assert(module != nullptr);
    std::cout << "Load model successful!" << std::endl;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::zeros({1,3,224,224}));
    at::Tensor output = module.forward(inputs).toTensor();
    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    std::cout << max_index << std::endl;
}

void Classfier(cv::Mat &image){
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255);
    torch::jit::script::Module module = torch::jit::load("../last.pt");  
    torch::Tensor output = module.forward({img_tensor}).toTensor();
    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    std::cout << max_index << std::endl;
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}

torch::jit::script::Module load_model(std::string model_path)
{
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error & e) {
        std::cerr << "error loading the model\n";
        exit(-1);
    }
    std::cout << "Load Finished !\n";
    return module;
}

// resize frame
cv::Mat resize_with_ratio(cv::Mat& img)   
{
    cv::Mat temImage;
    int w = img.cols;
    int h = img.rows;

    float t = 1.;
    float len = t * std::max(w, h);
    int dst_w = 640, dst_h = 640;
    cv::Mat image = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(128,128,128));
    cv::Mat imageROI;
    if(len==w)
    {
        float ratio = (float)h/(float)w;
        cv::resize(img,temImage,cv::Size(640,640*ratio),0,0,cv::INTER_LINEAR);
        imageROI = image(cv::Rect(0, ((dst_h-640*ratio)/2), temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    }
    else
    {
        float ratio = (float)w/(float)h;
        cv::resize(img,temImage,cv::Size(640*ratio,640),0,0,cv::INTER_LINEAR);
        imageROI = image(cv::Rect(((dst_w-640*ratio)/2), 0, temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    }

    return image;
}


typedef struct bbox
{
    int cla;         // 物体类别
    float conf;       // 置信度
    cv::Rect r;     //xyxy
} Box;


bool nms(const torch::Tensor& boxes, const torch::Tensor& scores, torch::Tensor &keep, int &count,float overlap, int top_k)
{
    count =0;
    keep = torch::zeros({scores.size(0)}).to(torch::kLong).to(scores.device());

    if(0 == boxes.numel())
    {
        return false;
    }
      
    torch::Tensor x1 = boxes.select(1,0).clone();
    torch::Tensor y1 = boxes.select(1,1).clone();
    torch::Tensor x2 = boxes.select(1,2).clone();
    torch::Tensor y2 = boxes.select(1,3).clone();
    torch::Tensor area = (x2-x1)*(y2-y1);
    // std::cout<<area<<std::endl;

    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(scores.unsqueeze(1), 0, 0);
    torch::Tensor v = std::get<0>(sort_ret).squeeze(1).to(scores.device());
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1).to(scores.device());

    int num_ = idx.size(0);
    if(num_ > top_k) //python:idx = idx[-top_k:]
    {
        idx = idx.slice(0,num_-top_k,num_).clone();
    }
    torch::Tensor xx1,yy1,xx2,yy2,w,h;
    while(idx.numel() > 0)
    {
        auto i = idx[-1].item().toInt();
        // std::cout<<"idx:"<<idx<<std::endl; 
        keep[count] = i;
        count += 1;
        if(1 == idx.size(0))
        {
            break;
        }
        idx = idx.slice(0,0,idx.size(0)-1).clone().reshape({idx.size(0)-1});
        // std::cout<<"idx:"<<idx<<std::endl;  

        xx1 = torch::index_select(x1, 0, idx);
        yy1 = torch::index_select(y1, 0, idx);
        xx2 = torch::index_select(x2, 0, idx);
        yy2 = torch::index_select(y2, 0, idx);

        xx1 = xx1.clamp(x1[i].item().toFloat(),INT_MAX*1.0);
        yy1 = yy1.clamp(y1[i].item().toFloat(),INT_MAX*1.0);
        xx2 = xx2.clamp(INT_MIN*1.0,x2[i].item().toFloat());
        yy2 = yy2.clamp(INT_MIN*1.0,y2[i].item().toFloat());

        w = xx2 - xx1;
        h = yy2 - yy1;

        w = w.clamp(0,INT_MAX);
        h = h.clamp(0,INT_MAX);

        torch::Tensor inter = w * h;
        torch::Tensor rem_areas = torch::index_select(area,0,idx);

        torch::Tensor union_ = (rem_areas - inter) + area[i];
        torch::Tensor Iou = inter * 1.0 / union_;
        torch::Tensor index_small = Iou < overlap;
        auto mask_idx = torch::nonzero(index_small).squeeze();
        idx = torch::index_select(idx,0,mask_idx);
    }
    return true;
}

int main(int argc, const char* argv[]) {

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1; 
    }

    cv::VideoCapture cap;
    cap.open(0);
    cv::Mat frame;
    if (!cap.read(frame)) {
        throw std::logic_error("Failed to get frame from cv::VideoCapture");
    }
    // cv::namedWindow("Smoke Detect", cv::WINDOW_AUTOSIZE);

    // Deserialize the ScriptModule from a file using torch::jit::load().
    auto module = load_model(argv[1]);
    float conf_thres = 0.4;

    // read frame
    // cv::Mat image;
    cv::Mat input;
    int delay = 33;
    
    while(cap.read(frame)){

        // frame = cv::imread("../000-0.jpg"); 
        // image.copyTo(frame) ;

        std::cout << "img size:" << frame.size()<< std::endl;           

        cv::Mat image(640, 640, frame.type()); 

        cv::resize(frame,image,image.size(),0,0, cv::INTER_LINEAR);
        // imshow("image", image);


        // image = resize_with_ratio(frame);
        

        cv::cvtColor(image, input, cv::COLOR_BGR2RGB);


        torch::Tensor tensor_image = torch::from_blob(input.data, {1,input.rows, input.cols,3}, torch::kByte).to(torch::kCPU);
        tensor_image = tensor_image.permute({0,3,1,2});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.div(255);
        int h = tensor_image.sizes()[2], w = tensor_image.sizes()[3];
          std::cout << h << std::endl;
          std::cout << w << std::endl;
          std::cout << tensor_image.sizes() << std::endl;
          


        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
        //infereing
        auto results = module.forward(inputs).toTuple();
    
        auto res = results->elements()[0].toTensor();
        // std::cout <<res <<std::endl;
        // std::cout << res.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

        //convent data

        auto xc =res[0].select(1, 4)>conf_thres;
        auto res_indexs = torch::nonzero(xc);

        torch::Tensor conf_res = torch::index_select(res, 1, res_indexs.reshape(res_indexs.numel()).toType(torch::kLong));
        std::cout << res_indexs.numel() << std::endl;

           if(res_indexs.numel() > 0){
            //    (center x, center y, width, height)
                std::cout << conf_res << std::endl;
                torch::Tensor cat_1 = conf_res[0];
                auto c4 =cat_1.select(1,4);
                auto c5 =cat_1.select(1,5);

                // std::cout << c4 << std::endl;
                // std::cout << c4[0].item().toFloat() <<","<<c4[1].item().toFloat()<< std::endl;

  
                double  gain = std::min(image.rows / (frame.rows/1.0000), image.cols / (frame.cols/1.0000) );
                double  pad0 = (image.cols - frame.cols * gain) / 2;
                double  pad1 = (image.rows - frame.rows * gain) / 2;

                std::cout << "gain:"<< gain << std::endl;
                std::cout << "pad0:"<< pad0 << std::endl;
                std::cout << "pad1:"<< pad1 << std::endl;

                torch::Tensor conv_res = torch::full_like(conf_res[0],1);
                for (size_t i = 0; i < c4.numel(); i++)
                {

                    conv_res[i][5] = c4[i].item().toFloat() * c5[i].item().toFloat();
                    conv_res[i][4] = c4[i].item().toFloat();
                    //convert cxcywh->xyxy
                    conv_res[i][0] = cat_1[i][0] - cat_1[i][2] / 2;
                    conv_res[i][1] = cat_1[i][1] - cat_1[i][3] / 2;
                    conv_res[i][2] = cat_1[i][0] + cat_1[i][2] / 2;
                    conv_res[i][3] = cat_1[i][1] + cat_1[i][3] / 2;
                    // conv_res[i][2] = cat_1[i][2] ;
                    // conv_res[i][3] = cat_1[i][3] ;

                    // //convert img1->img0
                    // conv_res[i][0] = (conv_res[i][0] - pad0) / gain;
                    // conv_res[i][2] = (conv_res[i][2] - pad0) / gain;
                    // conv_res[i][1] = (conv_res[i][1] - pad1) / gain;
                    // conv_res[i][3] = (conv_res[i][3] - pad1) / gain;
                    // //convert xyxy->xywh
                    // conv_res[i][2] = conv_res[i][2] - conv_res[i][0] / 2;
                    // conv_res[i][3] = conv_res[i][3] - conv_res[i][1] / 2;

                    conv_res[i][5] = 0;

                }
                // std::cout << conv_res << std::endl;
                // for (size_t i = 0; i < c4.numel(); i++)
                // {
                // //   convert img1->img0
                //     conv_res[i][0] = (conv_res[i][0] - pad0) / gain;
                //     conv_res[i][2] = (conv_res[i][2] - pad0) / gain;
                //     conv_res[i][1] = (conv_res[i][1] - pad1) / gain;
                //     conv_res[i][3] = (conv_res[i][3] - pad1) / gain;
                //     //convert xyxy->xywh
                //     // conv_res[i][2] = conv_res[i][2] - conv_res[i][0] / 2;
                //     // conv_res[i][3] = conv_res[i][3] - conv_res[i][1] / 2;
                // }
                  
                std::cout << conv_res << std::endl;
      

                if(conf_res.numel() > 0 ){



                    torch::Tensor boxes = conv_res.slice(1,0,4);
                    torch::Tensor scores = conv_res.slice(1,4,5);
                    // std::cout << "boxes:" << std::endl; 
                    // std::cout << boxes << std::endl; 
                    // std::cout << "scores:" << std::endl; 
                    // std::cout << scores << std::endl; 

                    torch::Tensor keep;
                    int count = 0;
                    nms(boxes, scores, keep, count, 0.5, 200);
                    // std::cout << "keep:" << std::endl; 
                    // std::cout << keep << std::endl; 
                    // std::cout << "count:" << std::endl; 
                    // std::cout << count << std::endl; 

                    torch::Tensor nms_index = torch::nonzero(keep);
                    // std::cout<<nms_index<<std::endl;


                    std::vector<bbox> boxs;
                    
                    for (size_t i = 0; i < nms_index.size(0); i++)
                    {

                        cv::Rect rect(conv_res[keep[i]][0].item().toFloat(),conv_res[keep[i]][1].item().toFloat(),conv_res[keep[i]][2].item().toFloat()-conv_res[keep[i]][0].item().toFloat(),conv_res[keep[i]][3].item().toFloat()-conv_res[keep[i]][1].item().toFloat());
                        Box box;
                        box.cla = conv_res[keep[i]][5].item().toFloat();
                        box.conf = conv_res[keep[i]][4].item().toFloat();
                        box.r = rect;

                        boxs.push_back(box);
                    }
                    std::cout<<" size : "<< boxs.size()<<std::endl;
                    for (size_t i = 0; i < boxs.size(); i++)
                    {
                        cv::rectangle(frame,  boxs[i].r, cv::Scalar(0, 0,250), 2);
                        cv::rectangle(image,  boxs[i].r, cv::Scalar(0, 0,250), 2);
                        cv::putText(image, "Smoke "+std::to_string(boxs[i].conf), cv::Point2f(boxs[i].r.x+2, boxs[i].r.y-2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 250));
                        

                    }

                

                    // cv::putText(frame, "Smoke", cv::Point2f(x1+20, y1), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 250));
                    // cv::circle(frame, cv::Point2f(x1,y1), 5, cv::Scalar(0, 255, 0), -1);
                    // cv::circle(frame, cv::Point2f(x2,y2), 8, cv::Scalar(0, 0, 250), -1);
                    // cv::rectangle(frame,  cv::Point2f(x1,y1), cv::Point2f(x2, y2), cv::Scalar(0, 0,250), 2);
                    // //  cv::rectangle(frame, head.rect, cv::Scalar(0, 255, 0), 2);
                }

                // cv::putText(image, "Smoke", cv::Point2f(0, 100), cv::FONT_HERSHEY_TRIPLEX, 2.0, cv::Scalar(0, 0, 250));

            }else{
                std::cout << "no res_indexs" << std::endl;
            }


        // cv::resizeWindow("Smoke Detect", 640, 640);
        cv::Mat img(frame.rows, frame.cols, image.type()); 
        cv::resize(image,img,img.size(),0,0, cv::INTER_LINEAR);
        cv::imshow("Smoke Detect",img);    //显示摄像头的数据
        // cv::imwrite('smo.jpg', image);
        // cv::imwrite("./smo.jpg", image);
        // cv::waitKey(0);
        cv::waitKey(30) ;
        int key = cv::waitKey(delay) & 255;
        if (key == 'p') {
            delay = (delay == 0) ? 33 : 0;
        } else if (key == 27) {
            break;
        }
    }

    
    // frame = cv::imread("../000-0.jpg");
    // image = resize_with_ratio(frame);
    // cv::cvtColor(image, input, CV_BGR2RGB); 


      
    // // cv.resizeWindow("Smoke Detect", 640, 640);
    // // cv::resizeWindow("Smoke Detect",640,640);
    // imshow("Smoke Detect",frame);
    // cv::waitKey(100000);


    return 0;
}


