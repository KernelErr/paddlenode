// The author is looking for a good job. :) (Kevin Li https://github.com/kernelerr)

#include <vector>
#include <node_api.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include "paddlelib.h"
#include "nodehelper.h"
using namespace paddle::lite_api;
using namespace cv;

PaddleLite paddlelite;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 224, 224};
const std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};

napi_value set_model_file(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1];
    char *model_file = new char[1001];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    if (argc != 1)
    {
        napi_throw_error(env, NULL, "Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype valuetype;
    NAPI_CALL(env, napi_typeof(env, args[0], &valuetype));
    if (valuetype != napi_string)
    {
        napi_throw_error(env, NULL, "Wrong argument type.");
        return NULL;
    }
    NAPI_CALL(env, napi_get_value_string_utf8(env, args[0], model_file, sizeof(char) * 1001, NULL));
    paddlelite.set_model_file(model_file);
    napi_value result = args[0];
    return result;
}

napi_value set_threads(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1];
    int32_t threads;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    if (argc != 1)
    {
        napi_throw_error(env, NULL, "Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype valuetype;
    NAPI_CALL(env, napi_typeof(env, args[0], &valuetype));
    if (valuetype != napi_number)
    {
        napi_throw_error(env, NULL, "Wrong argument type.");
        return NULL;
    }
    NAPI_CALL(env, napi_get_value_int32(env, args[0], &threads));
    paddlelite.set_threads(threads);
    napi_value result = args[0];
    return result;
}

napi_value set_power_mode(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1];
    int32_t powermode;
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    if (argc != 1)
    {
        napi_throw_error(env, NULL, "Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype valuetype;
    NAPI_CALL(env, napi_typeof(env, args[0], &valuetype));
    if (valuetype != napi_number)
    {
        napi_throw_error(env, NULL, "Wrong argument type.");
        return NULL;
    }
    NAPI_CALL(env, napi_get_value_int32(env, args[0], &powermode));
    paddlelite.set_power_mode(powermode);
    napi_value result = args[0];
    return result;
}

napi_value infer_float(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value args[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    if (argc != 2)
    {
        napi_throw_error(env, NULL, "Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype input_valuetype, shape_valuetype;
    NAPI_CALL(env, napi_typeof(env, args[0], &input_valuetype));
    NAPI_CALL(env, napi_typeof(env, args[1], &shape_valuetype));
    if (input_valuetype != napi_object || shape_valuetype != napi_object)
    {
        napi_throw_error(env, NULL, "Wrong argument type.");
        return NULL;
    }
    std::vector<int64_t> shape;
    uint32_t i, input_length, shape_length, input_size = 1;
    NAPI_CALL(env, napi_get_array_length(env, args[0], &input_length));
    NAPI_CALL(env, napi_get_array_length(env, args[1], &shape_length));
    for (i = 0; i < shape_length; i++)
    {
        napi_value e;
        int g;
        NAPI_CALL(env, napi_get_element(env, args[1], i, &e));
        NAPI_CALL(env, napi_get_value_int32(env, e, &g));
        shape.push_back(g);
        input_size *= g;
    }
    if (input_length != input_size)
    {
        napi_throw_error(env, NULL, "Input size doesn't match.");
        return NULL;
    }
    float *input_data = new float[input_size];
    for (i = 0; i < input_length; i++)
    {
        napi_value e;
        double g;
        NAPI_CALL(env, napi_get_element(env, args[0], i, &e));
        NAPI_CALL(env, napi_get_value_double(env, e, &g));
        *(input_data + i) = (float)g;
    }
    float *infer_res;
    infer_res = paddlelite.infer_float(input_data, shape);
    napi_value ret;
    NAPI_CALL(env, napi_create_array(env, &ret));
    for (i = 0; i < *infer_res - 1; i++)
    {
        napi_value e;
        NAPI_CALL(env, napi_create_double(env, *(infer_res + i + 1), &e));
        NAPI_CALL(env, napi_set_element(env, ret, i, e));
    }
    return ret;
}

napi_value image_file_classification(napi_env env, napi_callback_info info)
{
    char *image_path = new char[1001];
    double scalefactor;
    int size[2], mean[3];
    uint32_t size_length, mean_length, shape_length, i;
    bool swapRB;
    size_t argc = 6; // Image Path, shape, scalefactor, size, mean, swapRB
    napi_value args[6];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    if (argc != 6)
    {
        napi_throw_error(env, NULL, "Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype path_valuetype, shape_valuetype, scalefactor_valuetype, size_valuetype, mean_valuetype, swapRB_valuetype;
    NAPI_CALL(env, napi_typeof(env, args[0], &path_valuetype));
    NAPI_CALL(env, napi_typeof(env, args[1], &shape_valuetype));
    NAPI_CALL(env, napi_typeof(env, args[2], &scalefactor_valuetype));
    NAPI_CALL(env, napi_typeof(env, args[3], &size_valuetype));
    NAPI_CALL(env, napi_typeof(env, args[4], &mean_valuetype));
    NAPI_CALL(env, napi_typeof(env, args[5], &swapRB_valuetype));
    if (path_valuetype != napi_string)
    {
        napi_throw_error(env, NULL, "Wrong argument type.1");
        return NULL;
    }
    if (path_valuetype != napi_string || shape_valuetype != napi_object || scalefactor_valuetype != napi_number || size_valuetype != napi_object || mean_valuetype != napi_object || swapRB_valuetype != napi_boolean)
    {
        napi_throw_error(env, NULL, "Wrong argument type.");
        return NULL;
    }
    NAPI_CALL(env, napi_get_value_string_utf8(env, args[0], image_path, sizeof(char) * 1001, NULL));
    try
    {
        Mat input_image = imread(image_path);
        if (input_image.empty())
        {
            napi_throw_error(env, NULL, "Unable to read image.");
            return NULL;
        }
        NAPI_CALL(env, napi_get_value_double(env, args[2], &scalefactor));
        NAPI_CALL(env, napi_get_array_length(env, args[3], &size_length));
        NAPI_CALL(env, napi_get_array_length(env, args[4], &mean_length));
        if (size_length != 2 || mean_length != 3)
        {
            napi_throw_error(env, NULL, "Wrong array length.");
            return NULL;
        }
        for (i = 0; i < 2; i++)
        {
            napi_value e;
            NAPI_CALL(env, napi_get_element(env, args[3], i, &e));
            NAPI_CALL(env, napi_get_value_int32(env, e, &size[i]));
        }
        for (i = 0; i < 3; i++)
        {
            napi_value e;
            NAPI_CALL(env, napi_get_element(env, args[4], i, &e));
            NAPI_CALL(env, napi_get_value_int32(env, e, &mean[i]));
        }
        NAPI_CALL(env, napi_get_value_bool(env, args[5], &swapRB));
        Mat inputBlob = dnn::blobFromImage(input_image, scalefactor, Size(size[0], size[1]), Scalar(mean[0], mean[1], mean[2]), swapRB);
        std::vector<int64_t> shape;
        NAPI_CALL(env, napi_get_array_length(env, args[1], &shape_length));
        for (i = 0; i < shape_length; i++)
        {
            napi_value e;
            int g;
            NAPI_CALL(env, napi_get_element(env, args[1], i, &e));
            NAPI_CALL(env, napi_get_value_int32(env, e, &g));
            shape.push_back(g);
        }
        float *infer_res;
        infer_res = paddlelite.infer_float((float *)inputBlob.data, shape);
        napi_value ret;
        NAPI_CALL(env, napi_create_array(env, &ret));
        for (i = 0; i < *infer_res - 1; i++)
        {
            napi_value e;
            NAPI_CALL(env, napi_create_double(env, *(infer_res + i + 1), &e));
            NAPI_CALL(env, napi_set_element(env, ret, i, e));
        }
        return ret;
    }
    catch (const std::exception &e)
    {
        napi_throw_error(env, NULL, e.what());
        return NULL;
    }
}

#define DECLARE_NAPI_METHOD(name, func)         \
    {                                           \
        name, 0, func, 0, 0, 0, napi_default, 0 \
    }

napi_value Init(napi_env env, napi_value exports)
{
    napi_status status;
    napi_property_descriptor descriptors[] = {
        DECLARE_NAPI_METHOD("set_model_file", set_model_file),
        DECLARE_NAPI_METHOD("set_threads", set_threads),
        DECLARE_NAPI_METHOD("set_power_mode", set_power_mode),
        DECLARE_NAPI_METHOD("infer_float", infer_float),
        DECLARE_NAPI_METHOD("image_file_classification", image_file_classification)};
    status = napi_define_properties(env, exports, 5, descriptors);
    assert(status == napi_ok);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)