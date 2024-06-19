


#version 450
#extension GL_ARB_viewport_array : enable
layout (points) in;
layout (points, max_vertices = 1) out;//triangle_strip
const int num =  1;
layout(invocations = num) in;

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj[3];
} ubo;
layout(location = 0) in float hei[];
layout(location = 1) in vec4 pos[];


layout (location = 0) out vec4 fragColor;

////////////////////////////////////////////////////////////////////////////
float saturate (float x)
{
    return min(1.0, max(0.0,x));
}

vec3 saturate (vec3 x)
{
    return min(vec3(1.,1.,1.), max(vec3(0.,0.,0.),x));
}

vec3 spectral_jet(float w)
{

	float x = saturate((w - 0.0)/ 1.0);
	vec3 c;

	if (x < 0.25)
		c = vec3(0.0, 4.0 * x, 1.0);
	else if (x < 0.5)
		c = vec3(0.0, 1.0, 1.0 + 4.0 * (0.25 - x));
	else if (x < 0.75)
		c = vec3(4.0 * (x - 0.5), 1.0, 0.0);
	else
		c = vec3(1.0, 1.0 + 4.0 * (0.75 - x), 0.0);

	return saturate(c);
}
////////////////////////////////////////////////////////////////////////////
void main(void)
{	


	for(int i = 0; i < 1 ; i++)//
	{

		gl_Position =ubo.modelViewProj[2]*vec4(pos[i]);

		if (gl_InvocationID == 0)
		{
			gl_PointSize =float(5.0);
			
			fragColor = vec4(spectral_jet(hei[i]),1.0);

		}
		
		gl_ViewportIndex =gl_InvocationID;

		gl_PrimitiveID = gl_PrimitiveIDIn;
		
		EmitVertex();
	}
	
	EndPrimitive();
}
