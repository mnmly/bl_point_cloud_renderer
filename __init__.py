bl_info = {
    "name": "Point Cloud GL Renderer",
    "author": "Claude",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Point Cloud",
    "description": "Renders point clouds using OpenGL points and exports visualizations",
    "category": "3D View",
}

import bpy
import gpu
import random
import numpy as np
from bpy.props import (
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    StringProperty,
    PointerProperty,
)
from bpy.app.handlers import persistent
from bpy.types import Panel, Operator, PropertyGroup
from gpu_extras.batch import batch_for_shader


class PointCloudProperties(PropertyGroup):
    point_size: FloatProperty(
        name="Point Size",
        description="Size of the points",
        default=2.0,
        min=0.1,
        max=50.0,
    )
    
    point_color: FloatVectorProperty(
        name="Point Color",
        description="Color of the points",
        default=(1.0, 0.5, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )

    shader_type: EnumProperty(
        name="Shader Type",
        description="Type of shader to use",
        items=(
            ("BUILTIN", "Built-in", "Use built-in shader"),
            ("CUSTOM", "Custom", "Use custom shader"),
        ),
        default="BUILTIN",
    )

    use_random_colors: BoolProperty(
        name="Random Colors",
        description="Use random colors for points",
        default=False,
    )
    
    use_vertex_colors: BoolProperty(
        name="Vertex Colors",
        description="Use vertex colors for points",
        default=False,
    )

    update_on_frame_change: BoolProperty(
        name="Update on Frame Change",
        description="Update point cloud on frame change",
        default=False,
    )

    vertex_color_name: StringProperty(
        name="Vertex Colors",
        description="Vertex color attribute name",
        default="Col",
    )
    
    num_points: IntProperty(
        name="Number of Points",
        description="Number of points to generate",
        default=5000,
        min=10,
        max=1000000,
    )
    
    point_cloud_scale: FloatProperty(
        name="Scale",
        description="Scale of the point cloud",
        default=5.0,
        min=0.1,
        max=100.0,
    )
    
    import_file: StringProperty(
        name="Import File",
        description="File to import points from",
        default="",
        subtype='FILE_PATH',
    )
    
class PointCloudRenderSettings(PropertyGroup):
    use_transparent_background: BoolProperty(
        name="Transparent Background",
        description="Render with transparent background",
        default=False
    )
    background_color: FloatVectorProperty(
        name="Background Color",
        description="Color of the background",
        default=(0.0, 0.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )


class POINTCLOUD_OT_generate_points(Operator):
    bl_idname = "pointcloud.generate_points"
    bl_label = "Generate Points"
    bl_description = "Generate and render point cloud"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Remove existing handler if it exists
        remove_handler(context)
        
        props = context.scene.point_cloud_props
        
        # Generate points based on selected mode
        depsgraph = context.evaluated_depsgraph_get()
        visible_objects = [obj for obj in bpy.context.scene.objects if obj.visible_get()]
        points = []
        colors = []

        for obj in visible_objects:
            if obj.type == 'MESH':
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.data
                
                # Get vertices in world space
                world_matrix = obj.matrix_world
                _points = [world_matrix @ v.co for v in mesh.vertices]
                if props.use_random_colors:
                    _colors = [(random.uniform(0, 1), 
                            random.uniform(0, 1), 
                            random.uniform(0, 1), 
                            1.0) 
                            for _ in range(len(points))]
                elif props.use_vertex_colors and mesh.attributes.get(props.vertex_color_name):
                    color_values = [0.0] * (len(_points) * 4)
                    mesh.attributes[props.vertex_color_name].data.foreach_get("color_srgb", color_values)
                    _colors = [(color_values[i], color_values[i+1], color_values[i+2], color_values[i+3]) 
                                for i in range(0, len(color_values), 4)]
                else:
                    _colors = [props.point_color for _ in range(len(points))]
            else:
                continue
            
            points.extend(_points)
            colors.extend(_colors)

        # Create the point cloud handler
        create_point_cloud_handler(context, points, colors, props.point_size)
        
        return {'FINISHED'}

class POINTCLOUD_OT_clear_points(Operator):
    bl_idname = "pointcloud.clear_points"
    bl_label = "Clear Points"
    bl_description = "Clear the rendered point cloud"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        remove_handler(context)
        return {'FINISHED'}


class POINTCLOUD_OT_render_image(Operator):
    bl_idname = "pointcloud.render_image"
    bl_label = "Render Point Cloud"
    bl_description = "Render the point cloud to an image"
    
    def execute(self, context):
        # Make sure we have a point cloud
        global handler_batch
        if handler_batch is None:
            self.report({'ERROR'}, "No point cloud to render")
            return {'CANCELLED'}
        
        render_props = context.scene.point_cloud_render
        
        # Create or reuse an image
        resolution_x = context.scene.render.resolution_x
        resolution_y = context.scene.render.resolution_y
        
        image_name = "PointCloudRender"
        if image_name in bpy.data.images:
            image = bpy.data.images[image_name]
            image.scale(resolution_x, resolution_y)
        else:
            image = bpy.data.images.new(
                image_name, 
                width=resolution_x, 
                height=resolution_y,
                alpha=True
            )
        
        # Create off-screen buffer for rendering
        offscreen = gpu.types.GPUOffScreen(resolution_x, resolution_y)
        
        # Get current view matrix and projection matrix
        view_matrix = context.region_data.view_matrix
        projection_matrix = context.region_data.window_matrix

        camera = context.scene.camera
        
        # Get camera view matrix (inverse of the camera's world matrix)
        view_matrix = camera.matrix_world.inverted()
        
        # Get camera projection matrix
        render = context.scene.render
        aspect_ratio = resolution_x / resolution_y
        
        projection_matrix = camera.calc_matrix_camera(
            context.evaluated_depsgraph_get(),
            x=resolution_x,
            y=resolution_y,
            scale_x=render.pixel_aspect_x,
            scale_y=render.pixel_aspect_y,
        )

        # Draw to the offscreen buffer
        with offscreen.bind():
            # Clear the buffer
            fb = gpu.state.active_framebuffer_get()
            if render_props.use_transparent_background:
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            else:
                # Use the current viewport background color
                # theme = context.preferences.themes[0]
                bg_color = render_props.background_color
                fb.clear(color=(bg_color[0], bg_color[1], bg_color[2], bg_color[3]))
            
            # Set up the view
            gpu.matrix.load_matrix(view_matrix)
            gpu.matrix.load_projection_matrix(projection_matrix)
            
            # Set point size
            props = context.scene.point_cloud_props
            gpu.state.point_size_set(props.point_size)
            
            # Enable depth testing
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            
            # Draw the points
            global handler_shader, use_smooth_shader
            if not use_smooth_shader:
                handler_shader.uniform_float("color", props.point_color)
            handler_batch.draw(handler_shader)

            # Reset state
            gpu.state.point_size_set(1.0)
            
            # Read pixels
            buffer = fb.read_color(
                0, 0, resolution_x, resolution_y, 
                4, 0, 'UBYTE'
            )

        # Free the offscreen buffer
        offscreen.free()

        buffer.dimensions = resolution_x * resolution_y * 4
        image.pixels = [v / 255 for v in buffer]
        
        
        # Store in scene for UI access
        context.scene["pointcloud_render_image"] = image
        
        self.report({'INFO'}, "Point cloud rendered to image")
        return {'FINISHED'}


class POINTCLOUD_OT_save_render(Operator):
    bl_idname = "pointcloud.save_render"
    bl_label = "Save Rendered Image"
    bl_description = "Save the rendered point cloud image to a file"
    
    filepath: StringProperty(
        subtype='FILE_PATH',
        default="pointcloud.png"
    )
    
    filter_glob: StringProperty(
        default='*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp',
        options={'HIDDEN'}
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not context.scene.get("pointcloud_render_image"):
            self.report({'ERROR'}, "No rendered image to save")
            return {'CANCELLED'}
        
        image = context.scene["pointcloud_render_image"]
        
        # Set image format based on file extension
        file_ext = self.filepath.split(".")[-1].lower()
        if file_ext == "png":
            image.file_format = 'PNG'
        elif file_ext in ["jpg", "jpeg"]:
            image.file_format = 'JPEG'
        elif file_ext in ["tif", "tiff"]:
            image.file_format = 'TIFF'
        elif file_ext == "bmp":
            image.file_format = 'BMP'
        
        # Save image
        image.filepath_raw = self.filepath
        image.save()
        
        self.report({'INFO'}, f"Image saved to {self.filepath}")
        return {'FINISHED'}


class POINTCLOUD_PT_panel(Panel):
    bl_label = "Point Cloud Renderer"
    bl_idname = "POINTCLOUD_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Point Cloud'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.point_cloud_props
        
        # Point rendering settings
        layout.separator()
        layout.prop(props, "point_size")
        layout.prop(props, "use_random_colors")
        layout.prop(props, "use_vertex_colors")
        layout.prop(props, "shader_type")
        
        if not props.use_random_colors and not props.use_vertex_colors:
            layout.prop(props, "point_color")
        if props.use_vertex_colors:
            layout.prop(props, "vertex_color_name")

        
        layout.separator()
        layout.prop(props, "update_on_frame_change")
        
        # Operators
        layout.separator()
        row = layout.row()
        row.operator("pointcloud.generate_points", icon='POINTCLOUD_DATA')
        row.operator("pointcloud.clear_points", icon='X')


class POINTCLOUD_PT_render_panel(Panel):
    bl_label = "Point Cloud Render"
    bl_idname = "POINTCLOUD_PT_render_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Point Cloud'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        render_props = context.scene.point_cloud_render
        layout.prop(render_props, "use_transparent_background")
        if not render_props.use_transparent_background:
            layout.prop(render_props, "background_color")
        layout.separator()
        layout.operator("pointcloud.render_image", icon='RENDER_STILL')
        
        if context.scene.get("pointcloud_render_image"):
            layout.separator()
            box = layout.box()
            box.label(text="Rendered Image")
            # box.template_ID(context.scene, "pointcloud_render_image", new="image.new", open="image.open")
            box.operator("pointcloud.save_render", icon='FILE_TICK')


# Global variables to track handlers
draw_handler = None
handler_batch = None
handler_shader = None
use_smooth_shader = False

# Handler for file load events
def on_file_load(dummy):
    # Wait a bit to ensure the scene is fully loaded
    bpy.app.timers.register(generate_on_load, first_interval=0.5)

def generate_on_load():
    # Generate the point cloud if context is available
    try:
        bpy.ops.pointcloud.generate_points()
    except Exception as e:
        print(f"Could not auto-generate point cloud: {e}")
    return None  # Don't repeat the timer

# Handler for frame change events
@persistent
def on_frame_change(scene):
    try:
        if bpy.context.scene.point_cloud_props.update_on_frame_change:
            bpy.ops.pointcloud.generate_points()
    except Exception as e:
        print(f"Could not generate point cloud on frame change: {e}")

def create_point_cloud_handler(context, points, colors, point_size):
    global draw_handler, handler_batch, handler_shader, use_smooth_shader
    
    # Remove existing handler if it exists
    remove_handler(context)
    
    # Determine which shader to use
    first_color = len(colors) > 0 and colors[0] or (1.0, 1.0, 1.0, 1.0)
    use_smooth_shader = not all(c == first_color for c in colors)
    
    if context.scene.point_cloud_props.shader_type == 'CUSTOM':
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC4', "vertColor")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
        shader_info.push_constant('FLOAT', "frameCount")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC4', "color")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "FragColor")

        vertex_shader = '''
            void main()
            {
                vertColor = float4(color.rg, color.b * (frameCount / 100.0f), color.a);
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0f);
            }
        '''
        fragment_shader = '''
            void main()
            {
                FragColor = vertColor;
            }
        '''

        shader_info.vertex_source(vertex_shader)
        shader_info.fragment_source(fragment_shader)

        shader = gpu.shader.create_from_info(shader_info)
        del vert_out
        del shader_info
        handler_batch = batch_for_shader(shader, 'POINTS', {
            "pos": points,
            "color": colors
        })
        
    elif use_smooth_shader:
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        handler_batch = batch_for_shader(shader, 'POINTS', {
            "pos": points,
            "color": colors
        })
    else:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        handler_batch = batch_for_shader(shader, 'POINTS', {
            "pos": points
        })
    
    handler_shader = shader
    
    def draw():
        # Set point size
        gpu.state.point_size_set(point_size)
        
        # Enable depth testing
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        
        if context.scene.point_cloud_props.shader_type == 'CUSTOM':
            # For custom shader, point size is handled in the shader
            handler_shader.uniform_float("frameCount", context.scene.frame_current)
        elif not use_smooth_shader:
            handler_shader.uniform_float("color", first_color)

        handler_batch.draw(handler_shader)
        
        # Reset state
        gpu.state.point_size_set(1.0)
    
    # Add the draw handler
    draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')


def remove_handler(context):
    global draw_handler, handler_batch, handler_shader
    
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
        draw_handler = None
        handler_batch = None
        handler_shader = None
        
        # Request redraw to clear the points
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


classes = (
    PointCloudProperties,
    PointCloudRenderSettings,
    POINTCLOUD_OT_generate_points,
    POINTCLOUD_OT_clear_points,
    POINTCLOUD_OT_render_image,
    POINTCLOUD_OT_save_render,
    POINTCLOUD_PT_panel,
    POINTCLOUD_PT_render_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.point_cloud_props = PointerProperty(type=PointCloudProperties)
    bpy.types.Scene.point_cloud_render = PointerProperty(type=PointCloudRenderSettings)
    
    # Register the load_post handler
    bpy.app.handlers.load_post.append(on_file_load)
    
    # Register the frame_change_post handler
    bpy.app.handlers.frame_change_pre.append(on_frame_change)


def unregister():
    # Remove the load_post handler
    if on_file_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_file_load)
    
    # Remove the frame_change_pre handler
    if on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(on_frame_change)
    
    # Remove any active handlers
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    if hasattr(bpy.types.Scene, "point_cloud_props"):
        del bpy.types.Scene.point_cloud_props
    
    if hasattr(bpy.types.Scene, "point_cloud_render"):
        del bpy.types.Scene.point_cloud_render


if __name__ == "__main__":
    register()