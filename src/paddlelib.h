#ifndef _PADDLELITE_LIB_
#define _PADDLELITE_LIB_
#include <iostream>
#include <vector>
#include "paddle_api.h"

namespace paddle
{
    namespace lite_api
    {
        class PaddleLite
        {
        private:
            MobileConfig config;
            std::shared_ptr<PaddlePredictor> predictor;
            int64_t ShapeProduction(const shape_t &shape);

        public:
            PaddleLite();
            ~PaddleLite();
            void set_model_file(std::string model_name);
            void set_threads(int);
            void set_power_mode(int);
            float* infer_float(float* input_data, std::vector<int64_t> shape);
        };
    } // namespace lite_api
} // namespace paddle
#endif