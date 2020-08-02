#include <numeric>
#include <iostream>
#include <memory>
#include <chrono>


#define GOOGLE_GLOG_DLL_DECL
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"
#include "paddlelib.h"

using paddle::AnalysisConfig;
using namespace paddle;

PaddleInference::PaddleInference()
{
}

PaddleInference::~PaddleInference()
{
}

void PaddleInference::set_combined_model(std::string model_dir)
{
    config.SetModel(model_dir);
    config.SwitchUseFeedFetchOps(false);
    config.EnableMKLDNN();
    config.EnableMemoryOptim();
    predictor = CreatePaddlePredictor(config);
}

float *PaddleInference::infer_float(float *input_data, const std::vector<int>& input_shape)
{
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputTensor(input_names[0]);
    input_t->Reshape(input_shape);
    input_t->copy_from_cpu<float>(input_data);

    CHECK(predictor->ZeroCopyRun());

    std::vector<std::string> out_names = predictor->GetOutputNames();
    std::unique_ptr<ZeroCopyTensor> output_t = predictor->GetOutputTensor(out_names[0]);
    std::vector<float> out_data;
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->copy_to_cpu(out_data.data());
    out_data.insert(out_data.begin(), out_num);

    float *output_data = new float[out_data.size()];
    std::copy(out_data.begin(), out_data.end(), output_data);
    return output_data;
}