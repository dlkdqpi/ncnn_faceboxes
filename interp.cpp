// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "interp.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Interp);

Interp::Interp()
{
    one_blob_only = false;
}

int Interp::load_param(const ParamDict& pd)
{
    resize_type = pd.get(0, 0);
    height_scale = pd.get(1, 1.f);
    width_scale = pd.get(2, 1.f);
    output_height = pd.get(3, 0);
    output_width = pd.get(4, 0);

    return 0;
}

void caffe_cpu_interp2(const int channels,
    const float *data1, const int height1, const int width1,
    float *data2, const int height2, const int width2) {

    // special case: just copy
    if (height1 == height2 && width1 == width2) {
        for (int h2 = 0; h2 < height2; ++h2) {
            const int h1 = h2;
            for (int w2 = 0; w2 < width2; ++w2) {
                const int w1 = w2;
                const float* pos1 = &data1[h1 * width1 + w1];
                float* pos2 = &data2[h2 * width2 + w2];
                for (int c = 0; c < channels; ++c) {
                    pos2[0] = pos1[0];
                    pos1 += width1 * height1;
                    pos2 += width2 * height2;
                }
                
            }
        }
        return;
    }
    const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
    const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
    for (int h2 = 0; h2 < height2; ++h2) {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = float(1.) - h1lambda;
        for (int w2 = 0; w2 < width2; ++w2) {
            const float w1r = rwidth * w2;
            const int w1 = w1r;
            const int w1p = (w1 < width1 - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = float(1.) - w1lambda;
            const float* pos1 = &data1[h1 * width1 + w1];
            float* pos2 = &data2[h2 * width2 + w2];
            for (int c = 0; c < channels; ++c) {
                pos2[0] =
                    h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                    h1lambda * (w0lambda * pos1[h1p * width1] + w1lambda * pos1[h1p * width1 + w1p]);
                pos1 += width1 * height1;
                pos2 += width2 * height2;
            }
            
        }
    }
}

int Interp::forward(const std::vector<Mat> &bottom_blob, std::vector<Mat> &top_blob, const Option& opt) const
{
    int src_h = bottom_blob[0].h;
    int src_w = bottom_blob[0].w;
    int src_c = bottom_blob[0].c;
    size_t elemsize = bottom_blob[0].elemsize;

    int dst_h = bottom_blob[1].h;
    int dst_w = bottom_blob[1].w;
    
    top_blob[0].create(dst_w, dst_h, src_c, elemsize, opt.blob_allocator);
    if (top_blob[0].empty())
        return -100;

    #pragma omp parallel for
    for (int q = 0; q < src_c; q++)
    {
        const float* ptr = bottom_blob[0].channel(q);
        float* outptr = top_blob[0].channel(q);
        caffe_cpu_interp2(1, ptr, src_h, src_w, outptr, dst_h, dst_w);
    }
    return 0;
}


} // namespace ncnn
