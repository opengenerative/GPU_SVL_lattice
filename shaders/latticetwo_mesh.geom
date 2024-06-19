

#version 450

#extension GL_ARB_viewport_array : enable

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

const int num =  1;
layout(invocations = num) in;

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj[5];
} ubo;

//push constants block
layout( push_constant ) uniform constants
{
	vec4 eyes;
} PushConstants;

layout(location = 0) in vec4 posit[];
layout(location = 1) in vec4 norm[];

layout (location = 0) out vec3 fragColor;



void main(void)
{	

	for(int i = 0; i < 3 ; i++)//
	{

		
		vec3 n_normal = normalize(norm[i].xyz*1.0);
	
		vec3 lightvector = normalize(PushConstants.eyes.xyz - posit[i].xyz);
		
		vec3 lightcolor = vec3(1.0,1.0,1.0);
	
		float amg = abs(dot(lightvector.xyz,n_normal.xyz));
		
		if (gl_InvocationID == 0)
			{
				gl_Position =ubo.modelViewProj[4]*posit[i];
				
				fragColor = lightcolor*amg;
				
				gl_PointSize =float(5);

			}


		gl_ViewportIndex =gl_InvocationID;

		gl_PrimitiveID = gl_PrimitiveIDIn;
		EmitVertex();
	}
	
	EndPrimitive();

}
