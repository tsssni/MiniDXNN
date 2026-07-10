/*!
  \file composite.frag
  \author Sho Ikeda
  \brief Composites ImGui overlay onto a scene texture without gamma correction
  \copyright Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

  SPDX-License-Identifier: MIT
*/

Texture2D g_SceneBuffer;
Texture2D g_ImGuiBuffer;
float2 g_Resolution;

float4 main(in float4 pos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_Target
{
    int3 pixel = int3(texcoord * g_Resolution, 0);
    float4 imgui = g_ImGuiBuffer.Load(pixel);
    float3 scene = g_SceneBuffer.Load(pixel).xyz;
    float3 output = lerp(imgui.rgb, scene, imgui.a);
    return float4(output, 1.0f);
}
