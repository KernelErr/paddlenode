#include <iostream>
#include <vector>
#include "paddle_api.h"
#include "paddlelib.h"

namespace paddle
{
    namespace lite_api
    {
        PaddleLite::PaddleLite()
        {
        }

        PaddleLite::~PaddleLite()
        {
        }

        void PaddleLite::set_model_file(std::string model_name)
        {
            config.set_model_from_file(model_name);
        }

        int64_t PaddleLite::ShapeProduction(const shape_t &shape)
        {
            int64_t res = 1;
            for (auto i : shape)
                res *= i;
            return res;
        }

        float* PaddleLite::infer_float(float* input_data, std::vector<int64_t> shape)
        {
            predictor = CreatePaddlePredictor<MobileConfig>(config);
            std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
            input_tensor->Resize(shape);
            auto *data = input_tensor->mutable_data<float>();
            int64_t input_shape = ShapeProduction(input_tensor->shape());
            for (int i = 0; i < input_shape; ++i)
            {
                data[i] = *(input_data + i);
            }
            predictor->Run();

            std::unique_ptr<const Tensor> output_tensor(
                std::move(predictor->GetOutput(0)));
            int64_t output_shape = ShapeProduction(output_tensor->shape());
            float* output_data = new float[output_shape+1];
            output_data[0] = output_tensor->shape()[1];
            for (int i = 0; i < output_shape; i ++){
                output_data[i+1] = output_tensor->data<float>()[i];
            }
            return output_data;
        }
    } // namespace lite_api
} // namespace paddle