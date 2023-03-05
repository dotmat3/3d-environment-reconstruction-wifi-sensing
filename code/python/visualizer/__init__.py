from IPython.core.display import HTML

import trimesh
from tqdm import tqdm

import base64
import os

def get_visualizer_template():
  _pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))
  template = open(os.path.join(_pwd, "index.html"), "r")
  html_data = template.read()
  template.close()

  return html_data

def get_visualizer_html(array, height=500):
  """Use the Three.js library in order to show the provided `array`."""

  html_data = get_visualizer_template()

  if array.sum() != 0:
    encoded = base64.b64encode(export_data(array)).decode("utf-8")
    result = html_data.replace("$B64GLTF", encoded)
  else:
    result = html_data

  result = result.replace('"', '&quot;')
  code = f'<style>body {{ padding: 7px 7px 10px 7px; }}</style><div><iframe srcdoc="{result}" width="100%" height="{height}px" style="border: none;"></iframe></div>'

  return code

def show(array, height=500):
  """Use the Three.js library in order to show the provided `array`."""
  return HTML(get_visualizer_html(array, height))


def get_grid_visualizer_html(data, rows, cols, height=500):
  item_height = height / rows

  num_data_items = len(data)

  style = "<style>body { padding: 7px 7px 10px 7px; }</style>"
  grid_items = f""

  for row in range(rows):
    for col in range(cols):
      index = row * cols + col

      if index < num_data_items:
        array = data[index]

        html_data = get_visualizer_template()

        if array.sum() != 0:
          encoded = base64.b64encode(export_data(array)).decode("utf-8")
          result = html_data.replace("$B64GLTF", encoded)
        else:
          result = html_data
        result = result.replace('"', '&quot;')
        
        iframe_src = result
      else:
        iframe_src = ""
      
      item = f'<div><iframe srcdoc="{iframe_src}" width="100%" height="{item_height}px" style="border: none;"></iframe></div>'

      grid_items += item

  grid_template_rows = "1fr " * rows
  grid_template_columns = "1fr " * cols
  grid_style = f"display: grid; grid-template-rows: {grid_template_rows}; grid-template-columns: {grid_template_columns}"

  code = f"""{style} <div style="{grid_style}">{grid_items}</div>"""

  return code


def show_grid(data, rows, cols, height=500):
  return HTML(get_grid_visualizer_html(data, rows, cols, height))


def export_data(data, file_type="glb"):
  # if the data is empty
  if data.sum() == 0:
    raise Exception("Empty data!")

  grid = trimesh.voxel.base.VoxelGrid(data)

  mesh = grid.as_boxes()
  scene = mesh.scene()
  scene.camera

  return scene.export(file_type="glb")

def get_diff_visualizer_html(src, target, height=500):
  html_data = get_visualizer_template()

  if src.sum() != 0:
    scene = trimesh.Scene()
    added_to_scene = False

    red = src & ~target
    if red.sum() != 0:
      red_grid = trimesh.voxel.base.VoxelGrid(red)
      red_mesh = red_grid.as_boxes(colors=[255, 0, 0])
      scene.add_geometry(red_mesh)
      added_to_scene = True

    green = src & target
    if green.sum() != 0:
      green_grid = trimesh.voxel.base.VoxelGrid(green)
      green_mesh = green_grid.as_boxes(colors=[0, 255, 0])
      scene.add_geometry(green_mesh)
      added_to_scene = True

    blue = ~src & target
    if blue.sum() != 0:
      blue_grid = trimesh.voxel.base.VoxelGrid(blue)
      blue_mesh = blue_grid.as_boxes(colors=[0, 0, 255])
      scene.add_geometry(blue_mesh)
      added_to_scene = True

    if added_to_scene:
      scene.camera
      data = scene.export(file_type="glb")

      encoded = base64.b64encode(data).decode("utf-8")
      result = html_data.replace("$B64GLTF", encoded)
    else:
      result = html_data  
  else:
    result = html_data
  
  result = result.replace('"', '&quot;')

  code = f'<style>body {{ padding: 7px 7px 10px 7px; }}</style><div><iframe srcdoc="{result}" width="100%" height="{height}px" style="border: none;"></iframe></div>'

  return code

def show_diff(src, target, height=500):
  return HTML(get_diff_visualizer_html(src, target, height))


def show_animation(data, frame_duration=500, camera_distance=20, height=500):
  encoded_data = list()

  for sample in tqdm(data):
    if sample.sum() != 0:
      encoded = base64.b64encode(export_data(sample)).decode("utf-8")
      encoded_data.append(encoded)
    else:
      # This indicates an empty object
      encoded_data.append("$B64GLTF")

  print("Data processed!")

  html_data = get_visualizer_template()

  # Set the data
  result = html_data.replace("\"$B64GLTF\"", str(encoded_data))

  # Set frame duration and camera distance
  result = result.replace("\"$FRAME_DURATION\"", str(frame_duration))
  result = result.replace("\"$CAMERA_DISTANCE\"", str(camera_distance))

  result = result.replace('"', '&quot;')

  code = f'<style>body {{ padding: 7px 7px 10px 7px; }}</style><div><iframe srcdoc="{result}" width="100%" height="{height}px" style="border: none;"></iframe></div>'

  return HTML(code)