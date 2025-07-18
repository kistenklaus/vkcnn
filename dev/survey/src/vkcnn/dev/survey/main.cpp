

#include "src/vkcnn/dev/survey/conv.hpp"
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/dev/utils/merian.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RSCKC8_NR_P2.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"

vkcnn::dev::survey::ConvSurvey
survey_conv(const merian::ContextHandle &context) {

  vkcnn::shaders::Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2
      mma16x8x8_chwc8_rcskc8_hr_p2;

  vkcnn::shaders::ConvTemplate *shaders[] = {
      &mma16x8x8_chwc8_rcskc8_hr_p2, //
  };
  using namespace vkcnn;
  using namespace vkcnn::dev::survey;

  ConvMatrix config = {
      .inout =
          {
              {1920, 1080, 3, 3},   //
              {1920, 1080, 3, 9},   //
              {1920, 1080, 9, 32},  //
              {1920, 1080, 32, 32}, //
              {1920, 1080, 73, 64}, //
              {1920, 1080, 64, 64}, //
              {1920, 1080, 64, 32}, //
              {1920, 1080, 32, 3},  //

              {960, 540, 32, 48},  //
              {960, 540, 48, 48},  //
              {960, 540, 128, 64}, //
              {960, 540, 64, 64},  //

              {480, 270, 64, 64},  //
              {480, 270, 160, 96}, //
              {480, 270, 96, 96},  //

              {240, 135, 80, 80},   //
              {240, 135, 144, 112}, //
              {240, 135, 112, 112}, //

              {120, 68, 96, 96}, //
          },
      .kernelSize =
          {
              glm::uvec2(3, 3) //
          },
      .types =
          {
              {FloatType::F16, FloatType::F16, FloatType::F16}, //
          },
      .layouts =
          {
              {ActivationLayout::CHWC8, ActivationLayout::CHWC8},
          },
      .activationFunctions = {
          std::nullopt, //
      }};

  return vkcnn::dev::survey::conv(context, shaders, config);
}

int main() {
  merian::ContextHandle context = vkcnn::merian::createContext();
  {
    auto survey = survey_conv(context);
    survey.print();
  }
}
