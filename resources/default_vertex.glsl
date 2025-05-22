#include "lygia/generative/cnoise.glsl"

void main()
{
    vertColor = float4(color.rg, color.b * (frameCount / 100.0f), color.a);
    vec3 position = pos;
    position.z += cnoise(vec3(position.xy * 2.01, frameCount / 100.0f));
    gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0f);
}
