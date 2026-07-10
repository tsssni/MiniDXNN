/*!
  \file display.frag
  \author Sho Ikeda
  \brief Fullscreen texture display fragment shader
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

Texture2D g_Texture;
SamplerState g_Sampler;
float2 g_TexelSize;

float4 main(in float4 pos : SV_Position) : SV_Target
{
    float2 uv = pos.xy * g_TexelSize;
    return g_Texture.Sample(g_Sampler, uv);
}
