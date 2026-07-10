/*!
  \file display.vert
  \author Sho Ikeda
  \brief Fullscreen triangle vertex shader for texture display
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

float4 main(in uint idx : SV_VertexID) : SV_Position
{
    return 1.0f - float4(4.0f * (idx & 1), 4.0f * (idx >> 1), 1.0f, 0.0f);
}
