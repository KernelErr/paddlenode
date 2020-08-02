#ifndef _PADDLELITE_LIB_
#define _PADDLELITE_LIB_
#include "paddle/include/paddle_inference_api.h"

using paddle::AnalysisConfig;

class PaddleInference
{
private:
    AnalysisConfig config;
    std::unique_ptr<paddle::PaddlePredictor> predictor;

public:
    PaddleInference();
    ~PaddleInference();
    void set_combined_model(std::string model_name);
    float *infer_float(float *input_data, const std::vector<int>& input_shape);
};
#endif