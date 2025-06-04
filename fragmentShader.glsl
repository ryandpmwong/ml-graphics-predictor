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



out vec4 fragmentColor;


vec3 fresnelSchick(vec3 r0, float cosAngle)
{
 	return r0 + (vec3(1.0) - r0) * pow(1.0 - cosAngle, 5.0);
}

void main() 
{
	// no lighting start
	/*
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;

	fragmentColor = vec4(materialDiffuse, material_alpha);
	*/
	// no lighting end

	// basic lighting start
	/*
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	fragmentColor = vec4(incommingLight, material_alpha);
	*/
	// basic lighting end

	// also basic lighting
	/*
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 outgoingLight = incommingLight * materialDiffuse;
	fragmentColor = vec4(outgoingLight, material_alpha);
	*/

	// add in ambient lighting
	/*
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 outgoingLight = (incommingLight + ambientLightColourAndIntensity) * materialDiffuse;
	fragmentColor = vec4(outgoingLight, material_alpha);
	*/

	// specular schmick  -  bad???
	/*
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 viewSpaceDirToEye = normalize(-v2f_viewSpacePosition);
	vec3 halfVector = normalize(viewSpaceDirToEye + viewSpaceDirToLight);
	float specularIntensity = pow(max(0.0, dot(halfVector, viewSpaceNormal)), material_specular_exponent);
	vec3 materialSpecular = texture(specular_texture, v2f_texCoord).xyz * material_specular_color;
	vec3 outgoingLight = (incommingLight + ambientLightColourAndIntensity) * materialDiffuse + incommingLight * specularIntensity * materialSpecular;
	fragmentColor = vec4(outgoingLight, material_alpha);
	*/

	// normalising the blinn phong whatever the schmick
	/*
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 viewSpaceDirToEye = normalize(-v2f_viewSpacePosition);
	vec3 halfVector = normalize(viewSpaceDirToEye + viewSpaceDirToLight);
	float specularNormalizationFactor = ((material_specular_exponent + 2.0) / (2.0));
	float specularIntensity = specularNormalizationFactor * pow(max(0.0, dot(halfVector, viewSpaceNormal)), material_specular_exponent);
	vec3 materialSpecular = texture(specular_texture, v2f_texCoord).xyz * material_specular_color;
	vec3 outgoingLight = (incommingLight + ambientLightColourAndIntensity) * materialDiffuse + incommingLight * specularIntensity * materialSpecular;
	fragmentColor = vec4(outgoingLight, material_alpha);
	*/

	// add in fresnel schmick
	/*
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 viewSpaceDirToEye = normalize(-v2f_viewSpacePosition);
	vec3 halfVector = normalize(viewSpaceDirToEye + viewSpaceDirToLight);
	float specularNormalizationFactor = ((material_specular_exponent + 2.0) / (2.0));
	float specularIntensity = specularNormalizationFactor * pow(max(0.0, dot(halfVector, viewSpaceNormal)), material_specular_exponent);
	vec3 materialSpecular = texture(specular_texture, v2f_texCoord).xyz * material_specular_color;
	vec3 fresnelSpecular = fresnelSchick(materialSpecular, max(0.0, dot(viewSpaceDirToLight, halfVector)));
	vec3 outgoingLight = (incommingLight + ambientLightColourAndIntensity) * materialDiffuse + incommingLight * specularIntensity * fresnelSpecular;
	fragmentColor = vec4(outgoingLight, material_alpha);
	*/

	// mirror ball
	/*
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	vec3 viewSpaceDirToEye = normalize(-v2f_viewSpacePosition);
	vec3 worldSpaceReflectionDir = viewToWorldRotationTransform * reflect(-viewSpaceDirToEye, viewSpaceNormal);
	vec3 envSample = texture(environmentCubeTexture, worldSpaceReflectionDir).xyz;
	fragmentColor = vec4(envSample, material_alpha);
	*/

	// mirror + fresno
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition- v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incommingLight = incomingIntensity * lightColourAndIntensity;
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 viewSpaceDirToEye = normalize(-v2f_viewSpacePosition);
	vec3 halfVector = normalize(viewSpaceDirToEye + viewSpaceDirToLight);
	float specularNormalizationFactor = ((material_specular_exponent + 2.0) / (2.0));
	float specularIntensity = specularNormalizationFactor * pow(max(0.0, dot(halfVector, viewSpaceNormal)), material_specular_exponent);
	vec3 materialSpecular = texture(specular_texture, v2f_texCoord).xyz * material_specular_color;
	vec3 fresnelSpecular = fresnelSchick(materialSpecular, max(0.0, dot(viewSpaceDirToLight, halfVector)));
	vec3 worldSpaceReflectionDir = viewToWorldRotationTransform * reflect(-viewSpaceDirToEye, viewSpaceNormal);
	vec3 envSample = texture(environmentCubeTexture, worldSpaceReflectionDir).xyz;
	vec3 fresnelSpecularEye = fresnelSchick(materialSpecular, max(0.0, dot(viewSpaceDirToEye, viewSpaceNormal)));
	vec3 outgoingLight = (incommingLight + ambientLightColourAndIntensity) * materialDiffuse + incommingLight * specularIntensity * fresnelSpecular + envSample * fresnelSpecularEye;
	fragmentColor = vec4(outgoingLight, material_alpha);
}
