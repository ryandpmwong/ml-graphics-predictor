#--- Imports ---#
#region
from OpenGL.GL import *
import math
import numpy as np
import time

import magic as magic
import lab_utils as lu
from ObjModel import ObjModel
import glob
import os

import graphics_utils as gu
#endregion

#--- Globals ---#
#region

# light
g_lightYaw = 25.0
g_lightYawSpeed = 0.0#145.0
g_lightPitch = -75.0
g_lightPitchSpeed = 0.0#30.0
g_lightDistance = 250.0
g_lightColourAndIntensity = lu.vec3(0.9, 0.9, 0.6)
g_ambientLightColourAndIntensity = lu.vec3(0.5)

# camera
g_zTranslation = 1000
g_yTranslation = 250
g_camera = lu.FreeCamera([0,g_yTranslation,g_zTranslation],180,0)
g_yFovDeg = 45.0

g_currentModelName = "shaderBall1.obj"
g_model = None
g_groundModel = None
g_wallModel = None
g_vertexShaderSource = ObjModel.defaultVertexShader
g_fragmentShaderSource = ObjModel.defaultFragmentShader
g_startTime = 0


g_level = 3

shaders = ['graphics/fragmentShader1.glsl', 'graphics/fragmentShader2.glsl', 'graphics/fragmentShader.glsl']
g_currentFragmentShaderName = shaders[g_level - 1]


# custom sphere parameters
g_use_custom_sphere = False
g_sphere_centre = [0.0, 250.0, 200.0]
g_radius = 75.0


g_currentEnvMapName = "Daylight"

g_environmentCubeMap = None

g_reloadTimeout = 1.0

g_currentMaterial = 0

my_counter = 0

DATA_PATH = "./cnn_data"

g_savedImageDir = f"{DATA_PATH}/test_data_1"
g_savedImageCounter = 0
g_maxImages = 0

"""
    Set the texture unit to use for the cube map to the next 
    free one (free as in not used by the ObjModel)
"""
TU_EnvMap = ObjModel.TU_Max
#endregion
#--- Callbacks ---#
#region
def update(dt: float, keys: dict[str, bool], 
           mouse_delta: list[float]) -> None:
    """
        Update the state of the world.

        Parameters:

            dt: frametime

            keys: current state of all keys

            mouse_delta: mouse movement since the last frame
    """
    global g_camera
    global g_reloadTimeout
    global g_lightYaw
    global g_lightYawSpeed
    global g_lightPitch
    global g_lightPitchSpeed

    g_camera.update(dt, keys, [0.0, 0.0])

def setDefaultUniformBindings(shaderProgram):
    assert glGetIntegerv(GL_CURRENT_PROGRAM) == shaderProgram

    glUniform1i(lu.get_uniform_location_debug(shaderProgram, "diffuse_texture"), ObjModel.TU_Diffuse);
    glUniform1i(lu.get_uniform_location_debug(shaderProgram, "opacity_texture"), ObjModel.TU_Opacity);
    glUniform1i(lu.get_uniform_location_debug(shaderProgram, "specular_texture"), ObjModel.TU_Specular);
    glUniform1i(lu.get_uniform_location_debug(shaderProgram, "normal_texture"), ObjModel.TU_Normal);

def render_frame(x_offset: int, width: int, height: int) -> None:
    """
        Draws a frame

        Parameters:

            x_offset: offset to render within the window
        
            width, height: the size of the frame buffer, or window
    """
    global g_camera
    global g_yFovDeg
    global g_model
    global my_counter
    global g_zTranslation
    global g_maxImages
    global g_savedImageCounter
    global g_groundModel
    global g_wallModel
    global g_level
    global g_startTime

    colour = np.array([1,1,0,1], np.float32)
    
    light_position = [0,2000,0]

    """
        This configures the fixed-function transformation from 
        Normalized Device Coordinates (NDC) to the screen 
        (pixels - called 'window coordinates' in OpenGL documentation).
        See: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glViewport.xhtml
    """
    glViewport(x_offset, 0, width, height)
    # Set the colour we want the frame buffer cleared to, 
    glClearColor(0.0, 0.2, 1.0, 1.0)
    """
        Tell OpenGL to clear the render target to the clear values 
        for both depth and colour buffers (depth uses the default)
    """
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

    world_to_view = g_camera.get_world_to_view_matrix(lu.vec3(0,1,0))
    view_to_clip = lu.make_perspective(g_yFovDeg, width/height, 0.1, 2500.0)

    model_to_view = world_to_view
    
    """
        This is a special transform that ensures that normal vectors 
        remain orthogonal to the surface they are supposed to be even
        in the prescence of non-uniform scaling. It is a 3x3 matrix 
        as vectors don't need translation anyway and this transform 
        is only for vectors, not points. If there is no non-uniform 
        scaling this is just the same as Mat3(modelToViewTransform)
    """
    model_to_view_normal = lu.inverse(
        lu.transpose(lu.Mat3(model_to_view)))

    """
        Bind the shader program such that we can set the uniforms 
        (model.render sets it again)
    """
    glUseProgram(g_shader)

    lu.set_uniform(g_shader, "viewSpaceLightPosition", 
                  lu.transform_point(world_to_view, light_position))
    lu.set_uniform(g_shader, "lightColourAndIntensity", 
                  g_lightColourAndIntensity)
    lu.set_uniform(g_shader, "ambientLightColourAndIntensity", 
                   g_ambientLightColourAndIntensity)
    """
        transform (rotate) light direction into view space 
        (as this is what the ObjModel shader wants)
    """
    lu.set_uniform(g_shader, "environmentCubeTexture", TU_EnvMap)
    lu.bind_texture(TU_EnvMap, g_environmentCubeMap, GL_TEXTURE_CUBE_MAP)
    lu.set_uniform(g_shader, "viewToWorldRotationTransform", lu.inverse(lu.Mat3(world_to_view)))
    """
        This dictionary contains a few transforms that are needed to 
        render the ObjModel using the default shader. It would be 
        possible to just set the modelToWorld transform, as this is 
        the only thing that changes between the objects, and compute 
        the other matrices in the vertex shader. However, this would 
        push a lot of redundant computation to the vertex shader and 
        makes the code less self contained, in this way we set all 
        the required parameters explicitly.
    """
    transforms = {
        "modelToClipTransform" : view_to_clip * world_to_view,
        "modelToViewTransform" : model_to_view,
        "modelToViewNormalTransform" : model_to_view_normal,
    }
    
    #g_model.render(g_shader, ObjModel.RF_Opaque, transforms)

    if my_counter == 0:
        g_model.fake_render(g_shader, ObjModel.RF_Opaque, transforms)
        my_counter = 1
    
    # ts stuff for transparent objects
    '''glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    flags = ObjModel.RF_Transparent| ObjModel.RF_AlphaTested
    g_model.render(g_shader, flags, transforms)
    glDisable(GL_BLEND)'''

    glUseProgram(g_shader)

    color_guy = [1.0, 0.3, 0.0]

    lu.set_uniform(g_shader, "material_diffuse_color", color_guy)
    lu.set_uniform(g_shader, "material_specular_color", [1.0, 0.3, 0.0])
    lu.set_uniform(g_shader, "material_emissive_color", [0.0, 0.0, 0.0])
    lu.set_uniform(g_shader, "material_alpha", 1.0)
    lu.set_uniform(g_shader, "material_specular_exponent", 55.0)


    '''
    // Textures set by OBJModel (names must be bound to the right texture unit, ObjModel.setDefaultUniformBindings helps with that.
    uniform sampler2D diffuse_texture;
    uniform sampler2D specular_texture;
    '''

    sphere_centre = [0,0,0]
    sphere_centre = [round(np.random.random() * 400 - 200, 4), round(np.random.random() * 400 + 100, 4), round(np.random.random() * 400 - 200, 4)]
    radius = round(np.random.random() * 50 + 25, 4)
    #sphere_centre = [(time.time() % 1) * 100,0,0]
    #radius = 100
    #sx, sz, small_length, large_length, sin_alpha, cos_alpha = gu.find_shadow_ellipse_point_source(light_position, sphere_centre, radius)
    #print(sx, sz, small_length, large_length, sin_alpha, cos_alpha, sphere_centre, radius)

    if g_use_custom_sphere:
        sphere_centre = g_sphere_centre
        radius = g_radius

    if g_level == 2 or g_level == 3:
        #sphere_centre = [50,250,(time.time() % 5) * 100 - 500]
        #radius = 50
        #sphere_centre = [0.0, 250.0, 0.0]
        sx, sz, small_length, large_length, sin_alpha, cos_alpha = gu.find_shadow_ellipse_point_source(light_position, sphere_centre, radius)
        #print(sx, sz, small_length, large_length, sin_alpha, cos_alpha, sphere_centre, radius)
    else:
        sx, sz, small_length, large_length, sin_alpha, cos_alpha = gu.find_shadow_ellipse_plane_source([1.0, -1.0, 0.0], sphere_centre, radius)
    
    lu.set_uniform(g_shader, "shadowCentreX", sx)
    lu.set_uniform(g_shader, "shadowCentreZ", sz-g_zTranslation)
    lu.set_uniform(g_shader, "shadowLength", large_length)
    lu.set_uniform(g_shader, "shadowWidth", small_length)
    lu.set_uniform(g_shader, "shadowCosAlpha", cos_alpha)
    lu.set_uniform(g_shader, "shadowSinAlpha", sin_alpha)

    #lu.my_draw_plane(view_to_clip, world_to_view, g_shader)
    lu.my_draw_sphere(sphere_centre, radius, colour, view_to_clip, world_to_view, g_shader, 4)

    if g_level == 3:
        g_groundModel.render(g_shader, ObjModel.RF_Opaque, transforms)
        g_wallModel.render(g_shader, ObjModel.RF_Opaque, transforms)
    else:
        lu.my_draw_plane(view_to_clip, world_to_view, g_shader)

    #lu.draw_sphere(light_position, 10.0, colour, view_to_clip, world_to_view)

    #lu.draw_sphere([0,0,0], 100.0, colour, view_to_clip, world_to_view)
    if g_savedImageCounter < g_maxImages:
        #file_name = f"saved_screens/new_test_folder/test_file_{g_savedImageCounter}.dat"
        #file_name = f"saved_screens/test_dataset/screen_{g_savedImageCounter}.dat"
        file_name = f"saved_screens/train_data/screen_{g_savedImageCounter}.dat"
        file_name = f"{g_savedImageDir}/screen_{g_savedImageCounter}.dat"
        gu.save_screen(width, height, file_name, sphere_centre, radius)

        # print progress
        if g_maxImages > 100 and g_savedImageCounter % (g_maxImages // 100) == 0:
            percentdone = g_savedImageCounter / g_maxImages
            if percentdone != 0 and percentdone != 1:
                print(percentdone)
                curtime = time.time() - g_startTime
                esttimeleft = curtime / percentdone * (1 - percentdone)
                print(f"taken {curtime}, estimated {esttimeleft} left")
        
        g_savedImageCounter += 1
        #gu.load_and_display_screen(file_name)

        if g_savedImageCounter == g_maxImages:
            print(f"{g_maxImages} images successfully saved.")

def init_resources() -> None:
    """
        Load any required resources.
    """
    global g_camera
    global g_lightDistance
    global g_shader
    global g_environmentCubeMap
    global g_startTime

    g_startTime = time.time()

    g_environmentCubeMap = lu.load_cubemap(
        "graphics_data/cube_maps/" + g_currentEnvMapName + "/%s.bmp", True)   
    load_model(g_currentModelName)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glEnable(GL_FRAMEBUFFER_SRGB)


    #self.defaultTextureOne = glGenTextures(1);
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1));
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_FLOAT, [1.0, 1.0, 1.0, 1.0]);

    #self.defaultNormalTexture = glGenTextures(1);
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1));
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, [0.5, 0.5, 0.5, 1.0]);
    glBindTexture(GL_TEXTURE_2D, 0);

    # Build with default first since that really should work, so then we have some fallback
    g_shader = build_shader(g_vertexShaderSource, g_fragmentShaderSource)

    reLoad_shader()    

    glUseProgram(g_shader)
    setDefaultUniformBindings(g_shader)
    glUseProgram(0)
#endregion
#--- Functions ---#
#region
def build_shader(vertex_src: str, fragment_src: str) -> int:
    """
        Build a shader.

        Parameters:

            vertex_src, fragment_src: source code for modules

        Returns:

            integer handle to the new shader program
    """
    shader = lu.build_shader(vertex_src, fragment_src, 
                             ObjModel.getDefaultAttributeBindings())
    if shader:
        glUseProgram(shader)
        ObjModel.setDefaultUniformBindings(shader)
        glUseProgram(0)
    return shader

'''def itemListCombo(currentItem, items, name):
    ind = items.index(currentItem)
    _,ind = imgui.combo(name, ind, items)
    return items[ind]'''

def reLoad_shader():
    global g_vertexShaderSource
    global g_fragmentShaderSource
    global g_shader
    
    vertexShader = ""
    with open('graphics/vertexShader.glsl') as f:
        vertexShader = f.read()
    fragmentShader = ""
    with open(g_currentFragmentShaderName) as f:
        fragmentShader = f.read()

    if g_vertexShaderSource != vertexShader \
        or fragmentShader != g_fragmentShaderSource:
        newShader = build_shader(vertexShader, fragmentShader)
        if newShader:
            if g_shader:
                glDeleteProgram(g_shader)
            g_shader = newShader
            print("Reloaded shader, ok!")
        g_vertexShaderSource = vertexShader
        g_fragmentShaderSource = fragmentShader

def load_model(modelName):
    global g_model
    global g_lightDistance
    global g_groundModel
    global g_wallModel
    g_model = ObjModel("graphics_data/" + modelName)
    #g_model = ObjModel("data/house.obj");
    g_groundModel = ObjModel("graphics_data/ground.obj")
    g_wallModel = ObjModel("graphics_data/wall.obj")

    #g_camera.target = g_model.centre
    #g_camera.distance = lu.length(g_model.centre - g_model.aabbMin) * 3.1
    g_lightDistance = lu.length(g_model.centre - g_model.aabbMin) * 1.3
#endregion
# This does all the openGL setup and window creation needed
# it hides a lot of things that we will want to get a handle on as time goes by.

def run(width, height):
    magic.run_program(
        "ML Graphics Predictor", 
        width, height, render_frame, init_resources, None, update)
    
#run(960, 960)
run(256, 256)