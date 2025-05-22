bl_info = {
    "name": "Point Cloud GL Renderer",
    "author": "Claude",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Point Cloud",
    "description": "Renders point clouds using OpenGL points and exports visualizations and animations",
    "category": "3D View",
}

import bpy
import gpu
import random
import numpy as np
import os
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

    vertex_shader_source: StringProperty(
        name="Vertex Shader Source",
        description="Name of the text block containing vertex shader code",
        default="vertex.glsl",
    )

    fragment_shader_source: StringProperty(
        name="Fragment Shader Source",
        description="Name of the text block containing fragment shader code",
        default="fragment.glsl",
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
    frame_start: IntProperty(
        name="Start Frame",
        description="First frame of the animation to render",
        default=1,
        min=0
    )
    frame_end: IntProperty(
        name="End Frame",
        description="Last frame of the animation to render",
        default=250,
        min=0
    )
    frame_step: IntProperty(
        name="Frame Step",
        description="Number of frames to skip between renders",
        default=1,
        min=1
    )
    frame_padding: IntProperty(
        name="Frame Padding",
        description="Number of digits to use for frame number in filename",
        default=4,
        min=1,
        max=8
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
            global handler_shader, use_smooth_shader, shader_type

            if shader_type == 'CUSTOM':
                # For custom shader, point size is handled in the shader
                handler_shader.uniform_float("frameCount", context.scene.frame_current)
            elif not use_smooth_shader:
                handler_shader.uniform_float("color", first_color)
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


class POINTCLOUD_OT_save_animation(Operator):
    bl_idname = "pointcloud.save_animation"
    bl_label = "Save Animation Frames"
    bl_description = "Render and save point cloud animation as a sequence of frames"
    
    directory: StringProperty(
        subtype='DIR_PATH',
        description="Directory to save the rendered frames"
    )
    
    filename_prefix: StringProperty(
        name="Filename Prefix",
        description="Prefix for the rendered frame filenames",
        default="frame_"
    )
    
    file_format: EnumProperty(
        name="File Format",
        description="Format to save the rendered frames",
        items=(
            ("PNG", "PNG", "Save as PNG"),
            ("JPEG", "JPEG", "Save as JPEG"),
            ("TIFF", "TIFF", "Save as TIFF"),
            ("BMP", "BMP", "Save as BMP"),
        ),
        default="PNG"
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        # Make sure we have a point cloud
        global handler_batch
        if handler_batch is None:
            self.report({'ERROR'}, "No point cloud to render")
            return {'CANCELLED'}
        
        render_props = context.scene.point_cloud_render
        original_frame = context.scene.frame_current
        
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        # Set up file extension
        if self.file_format == "PNG":
            file_ext = ".png"
        elif self.file_format == "JPEG":
            file_ext = ".jpg"
        elif self.file_format == "TIFF":
            file_ext = ".tiff"
        elif self.file_format == "BMP":
            file_ext = ".bmp"
        
        # Render each frame
        frames_rendered = 0
        self.report({'INFO'}, f"Starting animation render from frame {render_props.frame_start} to {render_props.frame_end}")
        
        try:
            for frame in range(render_props.frame_start, render_props.frame_end + 1, render_props.frame_step):
                # Set current frame
                context.scene.frame_set(frame)
                
                # Render the frame
                bpy.ops.pointcloud.render_image()
                
                if not context.scene.get("pointcloud_render_image"):
                    self.report({'ERROR'}, f"Failed to render frame {frame}")
                    continue
                
                # Get the rendered image
                image = context.scene["pointcloud_render_image"]
                
                # Set image format
                if self.file_format == "PNG":
                    image.file_format = 'PNG'
                elif self.file_format == "JPEG":
                    image.file_format = 'JPEG'
                elif self.file_format == "TIFF":
                    image.file_format = 'TIFF'
                elif self.file_format == "BMP":
                    image.file_format = 'BMP'
                
                # Generate filename with padded frame number
                frame_str = str(frame).zfill(render_props.frame_padding)
                filepath = os.path.join(self.directory, f"{self.filename_prefix}{frame_str}{file_ext}")
                
                # Save the image
                image.filepath_raw = filepath
                image.save()
                
                frames_rendered += 1
                
                # Update progress in the console
                print(f"Rendered frame {frame} ({frames_rendered}/{((render_props.frame_end - render_props.frame_start) // render_props.frame_step) + 1})")
        
        except Exception as e:
            self.report({'ERROR'}, f"Error rendering animation: {str(e)}")
            # Restore original frame
            context.scene.frame_set(original_frame)
            return {'CANCELLED'}
        
        # Restore original frame
        context.scene.frame_set(original_frame)
        
        self.report({'INFO'}, f"Animation saved: {frames_rendered} frames rendered to {self.directory}")
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

        if props.shader_type == 'CUSTOM':
            layout.prop(props, "vertex_shader_source")
            layout.prop(props, "fragment_shader_source")
        
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
        
        # Background settings
        box = layout.box()
        box.label(text="Background Settings")
        box.prop(render_props, "use_transparent_background")
        if not render_props.use_transparent_background:
            box.prop(render_props, "background_color")
        
        # Single image rendering
        layout.separator()
        box = layout.box()
        box.label(text="Single Image")
        box.operator("pointcloud.render_image", icon='RENDER_STILL')
        
        if context.scene.get("pointcloud_render_image"):
            box.operator("pointcloud.save_render", icon='FILE_TICK')
        
        # Animation rendering
        layout.separator()
        box = layout.box()
        box.label(text="Animation")
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(render_props, "frame_start")
        row.prop(render_props, "frame_end")
        col.prop(render_props, "frame_step")
        col.prop(render_props, "frame_padding")
        box.operator("pointcloud.save_animation", icon='RENDER_ANIMATION')


# Global variables to track handlers
draw_handler = None
handler_batch = None
handler_shader = None
use_smooth_shader = False
shader_type = 'UNIFORM_COLOR'

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
    global draw_handler, handler_batch, handler_shader, use_smooth_shader, shader_type
    
    # Remove existing handler if it exists
    remove_handler(context)
    
    # Determine which shader to use
    first_color = len(colors) > 0 and colors[0] or (1.0, 1.0, 1.0, 1.0)
    use_smooth_shader = not all(c == first_color for c in colors)
    props = context.scene.point_cloud_props
    shader_type = props.shader_type

    if props.shader_type == 'CUSTOM':
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC4', "vertColor")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
        shader_info.push_constant('FLOAT', "frameCount")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC4', "color")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "FragColor")

        if props.vertex_shader_source and props.vertex_shader_source in bpy.data.texts:
            vertex_shader = bpy.data.texts[props.vertex_shader_source].as_string()
        else:
            # Try to create the text block if it doesn't exist
            ensure_shader_text_blocks()
            if props.vertex_shader_source in bpy.data.texts:
                vertex_shader = bpy.data.texts[props.vertex_shader_source].as_string()
            else:
                vertex_shader = os.path.join(os.path.dirname(__file__), "resources", "default_vertex.glsl")

        if props.fragment_shader_source and props.fragment_shader_source in bpy.data.texts:
            fragment_shader = bpy.data.texts[props.fragment_shader_source].as_string()
        else:
            # Try to create the text block if it doesn't exist
            ensure_shader_text_blocks()
            if props.fragment_shader_source in bpy.data.texts:
                fragment_shader = bpy.data.texts[props.fragment_shader_source].as_string()
            else:
                fragment_shader = os.path.join(os.path.dirname(__file__), "resources", "default_fragment.glsl")

        shader_info.vertex_source(resolve_includes(vertex_shader, "third-party/lygia"))
        shader_info.fragment_source(resolve_includes(fragment_shader, "third-party/lygia"))

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
        
        if shader_type == 'CUSTOM':
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
    POINTCLOUD_OT_save_animation,
    POINTCLOUD_PT_panel,
    POINTCLOUD_PT_render_panel,
)


def ensure_shader_text_blocks():
    """Ensure that vertex.glsl and fragment.glsl text blocks exist, creating them from resources if needed."""
    # Check if vertex shader exists
    if "vertex.glsl" not in bpy.data.texts:
        # Create vertex shader from resource file
        vertex_path = os.path.join(os.path.dirname(__file__), "resources", "default_vertex.glsl")
        if os.path.exists(vertex_path):
            with open(vertex_path, 'r') as f:
                vertex_content = f.read()
            text_block = bpy.data.texts.new("vertex.glsl")
            text_block.write(vertex_content)
            print("Created vertex.glsl text block from resource file")
    
    # Check if fragment shader exists
    if "fragment.glsl" not in bpy.data.texts:
        # Create fragment shader from resource file
        fragment_path = os.path.join(os.path.dirname(__file__), "resources", "default_fragment.glsl")
        if os.path.exists(fragment_path):
            with open(fragment_path, 'r') as f:
                fragment_content = f.read()
            text_block = bpy.data.texts.new("fragment.glsl")
            text_block.write(fragment_content)
            print("Created fragment.glsl text block from resource file")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.point_cloud_props = PointerProperty(type=PointCloudProperties)
    bpy.types.Scene.point_cloud_render = PointerProperty(type=PointCloudRenderSettings)
    
    # Ensure shader text blocks exist
    # ensure_shader_text_blocks()
    
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


import os
import re
from pathlib import Path

def resolve_includes(shader_content, base_path="~/Downloads/lygia", current_file_dir=None, visited=None):
    """
    Recursively resolve #include directives in shader code.
    
    Args:
        shader_content (str): The shader code containing #include directives
        base_path (str): Base path where lygia is located
        current_file_dir (str): Directory of the current file being processed (for relative paths)
        visited (set): Set of already included files to prevent circular includes
    
    Returns:
        str: Shader code with all includes resolved
    """
    if visited is None:
        visited = set()
    
    # Expand the tilde in the path
    base_path = os.path.expanduser(base_path)
    
    # Pattern to match #include "path/to/file.glsl"
    include_pattern = r'#include\s*["\']([^"\']+)["\']'
    
    def replace_include(match):
        include_path = match.group(1)
        original_include_path = include_path
        
        # Determine the full file path based on include type
        if include_path.startswith('lygia/'):
            # Absolute path from lygia root: "lygia/math/mod289.glsl"
            include_path = include_path[6:]  # Remove 'lygia/' prefix
            full_path = os.path.join(base_path, include_path)
        elif include_path.startswith('../') or include_path.startswith('./'):
            # Relative path: "../math/mod289.glsl" or "./local.glsl"
            if current_file_dir:
                full_path = os.path.join(current_file_dir, include_path)
            else:
                # If no current file directory, treat as relative to base_path
                full_path = os.path.join(base_path, include_path)
        else:
            # Direct path without prefix: "math/mod289.glsl"
            full_path = os.path.join(base_path, include_path)
        
        # Normalize the path to resolve .. and . components
        full_path = os.path.normpath(full_path)
        
        # Convert to absolute path to handle circular includes properly
        abs_path = os.path.abspath(full_path)
        
        # Check for circular includes
        if abs_path in visited:
            return f"// Circular include detected: {original_include_path}"
        
        try:
            # Check if file exists
            if not os.path.exists(full_path):
                return f"// File not found: {original_include_path} (resolved to: {full_path})"
            
            # Read the included file
            with open(full_path, 'r', encoding='utf-8') as f:
                included_content = f.read()
            
            # Add to visited set
            visited.add(abs_path)
            
            # Get the directory of the included file for nested relative includes
            included_file_dir = os.path.dirname(full_path)
            
            # Recursively resolve includes in the included file
            resolved_content = resolve_includes(
                included_content, 
                base_path, 
                included_file_dir,  # Pass the directory of the included file
                visited.copy()
            )
            
            # Add comments to show what was included
            return f"// BEGIN INCLUDE: {original_include_path}\n{resolved_content}\n// END INCLUDE: {original_include_path}"
            
        except Exception as e:
            return f"// Error including {original_include_path}: {str(e)}"
    
    # Replace all includes
    resolved_content = re.sub(include_pattern, replace_include, shader_content)
    
    return resolved_content

if __name__ == "__main__":
    register()