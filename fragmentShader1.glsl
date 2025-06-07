#version 330

// Input from the vertex shader, will contain the interpolated (i.e., distance weighted average) vaule out put for each of the three vertex shaders that 
// produced the vertex data for the triangle this fragment is part of.
// Grouping the 'in' variables in this way makes OpenGL check that they match the vertex shader
in VertexData
{
	vec3 v2f_viewSpaceNormal;
	vec3 v2f_viewSpacePosition;
	vec2 v2f_texCoord;
};

// Material properties set by OBJModel.
uniform vec3 material_diffuse_color; 
uniform float material_alpha;
uniform vec3 material_specular_color; 
uniform vec3 material_emissive_color; 
uniform float material_specular_exponent;

// Textures set by OBJModel (names must be bound to the right texture unit, ObjModel.setDefaultUniformBindings helps with that.
uniform sampler2D diffuse_texture;
uniform sampler2D specular_texture;

// Other uniforms used by the shader
uniform vec3 viewSpaceLightPosition;
uniform vec3 lightColourAndIntensity;
uniform vec3 ambientLightColourAndIntensity;

// mirror reflection thingo
uniform samplerCube environmentCubeTexture;
uniform mat3 viewToWorldRotationTransform;


uniform float shadowCentreX;
uniform float shadowCentreZ;
uniform float shadowLength;
uniform float shadowWidth;
uniform float shadowCosAlpha;
uniform float shadowSinAlpha;



out vec4 fragmentColor;


vec3 fresnelSchick(vec3 r0, float cosAngle)
{
 	return r0 + (vec3(1.0) - r0) * pow(1.0 - cosAngle, 5.0);
}

float shadowVal(void)
{
	float dx = v2f_viewSpacePosition[0] - shadowCentreX;
	float dz = v2f_viewSpacePosition[2] - shadowCentreZ;
	float dist = pow((dz * shadowCosAlpha + dx * shadowSinAlpha) / shadowLength, 2) + pow((dz * shadowSinAlpha - dx * shadowCosAlpha) / shadowWidth, 2);
	if (dist <= 1) {
		return 1;
	}
	return 0;
}

void main() 
{
	// no lighting start
	
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;

	if (v2f_viewSpacePosition[1] == -250) {
		materialDiffuse = vec3(0.3, 1.0, 0.0);
	}

	fragmentColor = vec4(materialDiffuse, material_alpha);
	
	// no lighting end
}
