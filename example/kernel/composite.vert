/*!
  \file composite.vert
  \author Sho Ikeda
  \brief Fullscreen triangle vertex shader for compositing
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

VS_OUTPUT main(in uint idx : SV_VertexID)
{
    VS_OUTPUT output;
    output.texcoord = float2(1.0f - 2.0f * (idx & 1), 2.0f * (idx >> 1));
    output.pos = 1.0f - float4(4.0f * (idx & 1), 4.0f * (idx >> 1), 1.0f, 0.0f);
    return output;
}
