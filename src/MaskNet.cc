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

PyObject* ConvertImageToNumpy(const cv::Mat& image) {

    // Let's represent our 2D image with 3 channels
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};

    // 2 dimensions are for a 2D image, so let's add another dimension for channels
    return PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);

}

cv::Mat ConvertNumpyToMat2(PyObject* ndarr, int rows, int cols) {
    uchar *data = (uchar *)PyByteArray_AsString(ndarr);
    cv::Mat img(rows, cols, CV_8UC3, data);

    return img;
}

cv::Mat ConvertNumpyToMat(PyObject* numpyArray){
	// Convert PyObject to NumPy array if necessary
    if (!PyArray_Check(numpyArray)) {
        PyErr_SetString(PyExc_TypeError, "Input is not a numpy array");
        return cv::Mat();
    }

    // Get the NumPy array info
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(numpyArray);
    int ndims = PyArray_NDIM(array);
    npy_intp* shape = PyArray_SHAPE(array);
    auto type = PyArray_TYPE(array);
    int depth = CV_8U; // You would set the correct OpenCV depth based on the NumPy data type

    switch (type) {
        case NPY_UBYTE: depth = CV_8U; break;
        case NPY_BYTE: depth = CV_8S; break;
        case NPY_USHORT: depth = CV_16U; break;
        case NPY_SHORT: depth = CV_16S; break;
        case NPY_INT: depth = CV_32S; break;
        case NPY_FLOAT: depth = CV_32F; break;
        case NPY_DOUBLE: depth = CV_64F; break;
        default: return cv::Mat(); // Unsupported data type
    }

    // Check the number of dimensions
    if (ndims >= 2) {
        // Convert npy_intp* to int* for OpenCV Mat constructor
        std::vector<int> sizes(ndims);
        for (int i = 0; i < ndims; ++i) {
            sizes[i] = static_cast<int>(shape[i]);
        }

        // Create the cv::Mat object
        cv::Mat mat(ndims, sizes.data(), CV_MAKETYPE(depth, PyArray_DESCR(array)->elsize), PyArray_DATA(array));
        return mat;
    } else {
        PyErr_SetString(PyExc_ValueError, "Array has incorrect number of dimensions");
        return cv::Mat();
    }
}

SegmentDynObject::SegmentDynObject(){
    std::cout << "Importing Mask R-CNN Settings..." << std::endl;
    ImportSettings();
    std::string x;
    setenv("PYTHONPATH", this->py_path.c_str(), 1);
    x = getenv("PYTHONPATH"); // = /ws/external/src/python
    Py_Initialize();
    if(PyArray_API == NULL)
    {
	    import_array();
    }
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
    std::string name = "tmp.jpg";
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
    if (image.empty()){
        std::cerr << "The image is empty!" << std::endl;
    }
    std::cout << "GetSegmentation: " << dir+"/"+name  << std::endl;
    // OpenCV < 4.x
    // cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    // OpenCV == 4.x
    cv::Mat seg = cv::imread(dir+"/"+name, cv::IMREAD_UNCHANGED);
    //cv::Mat seg = cv::imread(dir+"/"+name, cv::IMREAD_GRAYSCALE);
    int rows = seg.rows;
    int cols = seg.cols;
    if(seg.empty()){
	//PyObject* py_image = this->cvt->toNDArray(image.clone());
	//PyObject* py_image = np.asarray(image.clone());
	//float *p = image.ptr<float>(rows);
	//npy_intp mdim[] = {rows, cols, 3};
	//PyObject* py_image = PyArray_SimpleNewFromData(2, mdim, NPY_UINT8, p);
	PyObject* py_image = ConvertImageToNumpy(image.clone());
	assert(py_image != NULL);
	if (py_image == NULL){
		std::cout << "py_image is NULL" << std::endl;
	}
	//PyObject_Print(py_image, stdout, 0);
	PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)",py_image);
	if (PyErr_Occurred()) {
 	   PyErr_Print();
	}
	//PyObject_Print(py_image, stdout, 0);
	// std::cout << py_mask_image << std::endl;
	if (py_mask_image == NULL)
		PyErr_Print();
	//seg = cvt->toMat(py_mask_image).clone();
	seg = ConvertNumpyToMat(py_mask_image).clone();
        seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground
        if(dir.compare("no_save")!=0){
            DIR* _dir = opendir(dir.c_str());
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

	    // Check the number of channels in the 'seg' image
	    if (seg.channels() != 1 && seg.channels() != 3 && seg.channels() != 4)
	    {
		// If the image is not 1, 3, or 4 channels, we need to convert it.
		cv::Mat converted_seg;
	    	if (seg.channels() == 2) {
			// Assuming the second channel is the mask, extract it.
			std::vector<cv::Mat> channels(2);
			cv::split(seg, channels);
			converted_seg = channels[1];
		} else {
			// For other numbers of channels, you may need a custom conversion
			// // Here we're assuming it's a multi-channel image and we're taking one channel.
			cv::extractChannel(seg, converted_seg, 0);
		}
		seg = converted_seg;
	    }
	    
	    // Now convert the image to CV_8U if it's not already
	    if (seg.depth() != CV_8U) {
		    seg.convertTo(seg, CV_8U);
	    }
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






















