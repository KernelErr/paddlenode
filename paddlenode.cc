// The author is looking for a good job. :) (Kevin Li https://github.com/kernelerr)

#include <vector>
#include <node_api.h>
#include <assert.h>
#include "paddlelib.h"
using namespace paddle::lite_api;

PaddleLite paddlelite;

napi_value set_model_file(napi_env env, napi_callback_info info)
{
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    char *model_file = new char[1001];
    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if(argc != 1){
        napi_throw_error(env,NULL,"Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype valuetype;
    status = napi_typeof(env, args[0], &valuetype);  
    if(valuetype != napi_string){
        napi_throw_error(env,NULL,"Wrong argument type.");
        return NULL;
    }
    status = napi_get_value_string_utf8(env, args[0], model_file, sizeof(char) * 1001, NULL);
    paddlelite.set_model_file(model_file);
    napi_value result = args[0];
    assert(status == napi_ok);
    return result;
}

napi_value infer_float(napi_env env, napi_callback_info info)
{
    napi_status status;
    size_t argc = 2;
    napi_value args[2];
    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if(argc != 2){
        napi_throw_error(env,NULL,"Check the amount of arguments.");
        return NULL;
    }
    napi_valuetype input_valuetype, shape_valuetype;
    status = napi_typeof(env, args[0], &input_valuetype);
    status = napi_typeof(env, args[1], &shape_valuetype);
    if(input_valuetype != napi_object || shape_valuetype != napi_object){
        napi_throw_error(env,NULL,"Wrong argument type.");
        return NULL;
    }
    std::vector<int64_t> shape;
    uint32_t i, input_length, shape_length, input_size=1;
    status = napi_get_array_length(env, args[0], &input_length);
    status = napi_get_array_length(env, args[1], &shape_length);
    for (i = 0; i < shape_length; i++)
    {
        napi_value e;
        int g;
        status = napi_get_element(env, args[1], i, &e);
        napi_get_value_int32(env, e, &g);
        shape.push_back(g);
        input_size *= g;
    }
    if(input_length != input_size){
        napi_throw_error(env,NULL,"Input size doesn't match.");
        return NULL;
    }
    float* input_data = new float[input_size];
    for (i = 0; i < input_length; i++)
    {
        napi_value e;
        double g;
        status = napi_get_element(env, args[0], i, &e);
        napi_get_value_double(env, e, &g);
        *(input_data + i) = (float)g;
    }
    float *infer_res;
    infer_res = paddlelite.infer_float(input_data, shape);
    napi_value ret;
    status = napi_create_array(env, &ret);
    for(i = 0; i < *infer_res; i ++){
        napi_value e;
        status = napi_create_double(env, *(infer_res + i), &e);
        status = napi_set_element(env, ret, i, e);
    }
    assert(status == napi_ok);
    return ret;
}

#define DECLARE_NAPI_METHOD(name, func)         \
    {                                           \
        name, 0, func, 0, 0, 0, napi_default, 0 \
    }

napi_value Init(napi_env env, napi_value exports)
{
    napi_status status;
    napi_property_descriptor set_model_file_desc = DECLARE_NAPI_METHOD("set_model_file", set_model_file);
    status = napi_define_properties(env, exports, 1, &set_model_file_desc);
    napi_property_descriptor infer_float_desc = DECLARE_NAPI_METHOD("infer_float", infer_float);
    status = napi_define_properties(env, exports, 2, &infer_float_desc);
    assert(status == napi_ok);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)