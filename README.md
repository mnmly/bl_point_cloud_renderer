# Blender Point Cloud GL Renderer

A Blender addon for simple point cloud rendering using `gpu` module.

## Features

- Real-time point cloud rendering directly in the 3D viewport
- Generate point clouds from mesh vertices
- Customizable point size and color
- Support for random colors or vertex color attributes
- Custom shader support with built-in and custom shader options
- Export visualizations as images
- Render animations as image sequences
- Dynamic updates during animation playback
- Transparent background support for renders
- Automatic point cloud regeneration when loading files or changing frames

## Installation

1. Download the latest release from the repository
2. In Blender, go to Edit > Preferences > Add-ons
3. Click "Install..." and select the downloaded ZIP file
4. Enable the addon by checking the box next to "3D View: Point Cloud GL Renderer"

## Usage

### Basic Operation

1. Open the Point Cloud panel in the 3D View sidebar (press N, then select the "Point Cloud" tab)
2. Adjust point cloud settings as desired:
   - Point Size: Controls the size of rendered points
   - Point Color: Sets the default color for points
   - Random Colors: Enable to use random colors for each point
   - Vertex Colors: Enable to use vertex color attributes from the mesh
3. Click "Generate Points" to create and display the point cloud
4. Use "Clear Points" to remove the point cloud from the viewport

### Rendering

1. Open the "Point Cloud Render" section in the panel
2. Configure render settings:
   - Transparent Background: Enable for transparent background in renders
   - Background Color: Set the background color when transparency is disabled
   - Frame settings for animations
3. Use "Render Image" to create a single image
4. Click "Save Render" to save the current render to disk
5. For animations, use "Save Animation" to render and save a sequence of frames

### Custom Shaders

1. Set "Shader Type" to "Custom"
2. The addon will create default vertex and fragment shader text blocks if they don't exist
3. Edit these shader blocks to customize the rendering
4. Shader changes will be applied when you regenerate the point cloud

## Event Handling

The addon includes event handlers that:
- Automatically regenerate the point cloud when loading a file
- Update the point cloud when animation frames change (when enabled)
