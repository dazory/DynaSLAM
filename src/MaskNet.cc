/**
* This file is part of DynaSLAM.
*
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/
#include <Python.h>
#include "MaskNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>

namespace DynaSLAM
{

#define U_SEGSt(a)\
    gettimeofday(&tvsv,0);\
    a = tvsv.tv_sec + tvsv.tv_usec/1000000.0
struct timeval tvsv;
double t1sv, t2sv,t0sv,t3sv;
void tic_initsv(){U_SEGSt(t0sv);}
void toc_finalsv(double &time){U_SEGSt(t3sv); time =  (t3sv- t0sv)/1;}
void ticsv(){U_SEGSt(t1sv);}
void tocsv(){U_SEGSt(t2sv);}
// std::cout << (t2sv - t1sv)/1 << std::endl;}

SegmentDynObject::SegmentDynObject(){
    std::cout << "Importing Mask R-CNN Settings..." << std::endl;
    ImportSettings();
    std::string x;
    setenv("PYTHONPATH", this->py_path.c_str(), 1);
    x = getenv("PYTHONPATH"); // = /ws/external/src/python
    Py_Initialize();
    std::cout << "Load module and instaces..." << std::endl;
    this->cvt = new NDArrayConverter();
    this->py_module = PyImport_ImportModule(this->module_name.c_str()); // MaskRCNN
    std::cout << "   py_module: " << this->module_name.c_str() << std::endl;
    assert(this->py_module != NULL);
    std::cout << "   class_name: " << this->class_name.c_str() << std::endl; // Mask
    this->py_class = PyObject_GetAttrString(this->py_module, this->class_name.c_str());
    assert(this->py_class != NULL);
    // python3 does not support PyInstance_New
    //this->net = PyInstance_New(this->py_class, NULL, NULL);
    this->net = PyObject_CallObject(this->py_class, NULL);
    assert(this->net != NULL);
    if(this->net == NULL)
        PyErr_Print();
    std::cout << "Creating net instance..." << std::endl;
    cv::Mat image  = cv::Mat::zeros(480,640,CV_8UC3); //Be careful with size!!
    std::cout << "Loading net parameters..." << std::endl;
    //std::string dir = "/ws/data/rgbd_dataset_freiburg1_xyz/rgb";
    //std::string name = "1305031102.175304.png";
    std::string dir = "/ws/external/masks";
    std::string name = "";
    GetSegmentation(image, dir, name);
    //PyErr_Print();
}

SegmentDynObject::~SegmentDynObject(){
    delete this->py_module;
    delete this->py_class;
    delete this->net;
    delete this->cvt;
}

cv::Mat SegmentDynObject::GetSegmentation(cv::Mat &image,std::string dir, std::string name){
	std::cout << "Hello GetSegmentation: " << dir+"/"+name  << std::endl;
    cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    //cv::Mat seg = cv::imread(dir+"/"+name, cv::IMREAD_GRAYSCALE);
    int rows = seg.rows;
    int cols = seg.cols;
    std::cout << "rows and cols:" << rows << ", " << cols << std::endl;
    if(seg.empty()){
	    std::cout << "No seg" << std::endl;
        PyObject* py_image = cvt->toNDArray(image.clone());
        assert(py_image != NULL);
	std::cout << "1 : " << this->get_dyn_seg.c_str() << std::endl;
	try{
        PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)",py_image);
	}catch(std::string error){
		std::cout << error << std::endl;
	PyErr_Print();
	}
        seg = cvt->toMat(py_mask_image).clone();
	std::cout << "2" << std::endl;
        seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground
	std::cout << "3" << std::endl;
        if(dir.compare("no_save")!=0){
            DIR* _dir = opendir(dir.c_str());
	    std::cout << "4" << std::endl;
            if (_dir) {closedir(_dir);}
            else if (ENOENT == errno)
            {
                const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (check == -1) {
                    std::string str = dir;
                    str.replace(str.end() - 6, str.end(), "");
                    mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                }
            }
	    std::cout << "5" << std::endl;
            cv::imwrite(dir+"/"+name,seg);
        }
    }
    return seg;
}

void SegmentDynObject::ImportSettings(){
    std::string strSettingsFile = "/ws/external/Examples/RGB-D/MaskSettings.yaml";
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);
    fs["py_path"] >> this->py_path;
    fs["module_name"] >> this->module_name;
    fs["class_name"] >> this->class_name;
    fs["get_dyn_seg"] >> this->get_dyn_seg;
    std::cout << "    py_path: "<< this->py_path << std::endl;
    std::cout << "    module_name: "<< this->module_name << std::endl;
    std::cout << "    class_name: "<< this->class_name << std::endl;
    std::cout << "    get_dyn_seg: "<< this->get_dyn_seg << std::endl;
}


}






















