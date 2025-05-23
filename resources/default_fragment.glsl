void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord) * 2.0;  // Scale to 0-1 range
    
    // Create a softer falloff
    float alpha = 1.0 - smoothstep(0.6, 1.0, dist);
    
    if (alpha < 0.01) {
        discard;
    }
    
    FragColor = vec4(vertColor.rgb, vertColor.a * alpha);
}