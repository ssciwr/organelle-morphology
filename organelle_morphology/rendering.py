import bpy
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import bpy
from dash import Dash, html, dcc, Input, Output
import tempfile
import base64
import dash_bootstrap_components as dbc
import os
import sys
import socket
from contextlib import redirect_stdout


def _export_meshes(
    p, filename: str, coloring="uniform", mcs_label=None, ids="*"
) -> Path:
    """Export all meshes of the given project as a .ply file.
    Coloring options:
    - uniform: All meshes are colored uniformly.
    - type: Meshes are colored by their type.
    - mcs: only the MCS parts are colored in and the rest is uniform.
    - mcs_type: MCS parts are colored in and the rest is colored by type.
    - curvature: Meshes are colored by their curvature.

    :param p: The project to export.
    :type p: Project
    :param filename: The filename to save the meshes to.
    :type filename: str
    :param coloring: Different color modes for the meshes.
    Possible values are "uniform", "type", "mcs", "mcs_type", "curvature", defaults to "uniform"
    :type coloring: str, optional
    :param mcs_label: The MCS label to color the meshes by, defaults to None
    :type mcs_label: str, optional
    :param ids: The filter ids to export, defaults to "*"
    :type ids: str, optional
    :return: The path to the saved file.
    :rtype: Path
    """

    if coloring not in ["uniform", "type", "mcs", "mcs_type", "curvature"]:
        raise ValueError(
            f"coloring must be one of 'uniform', 'type', 'mcs', 'mcs_type', 'curvature' but is {coloring}."
        )

    if coloring in ["mcs", "mcs_types"] and mcs_label is None:
        raise ValueError(
            "mcs_label must be provided when coloring by MCS or MCS types."
        )

    elif coloring in ["mcs", "mcs_type"] and mcs_label not in p._mcs_labels:
        raise ValueError(f"mcs_label {mcs_label} not found in project.")

    file_path = Path(filename)
    file_path.parents[0].mkdir(parents=True, exist_ok=True)
    meshes = []
    types = []
    orgs = []

    for org in p.organelles(ids):
        meshes.append(org.mesh)
        types.append(org.id.split("_")[0])
        orgs.append(org)

    if coloring == "uniform":
        # apply grey default color.
        _apply_uniform_color(meshes, (31, 119, 180, 255))
    elif coloring == "type":
        _apply_type_color(meshes, types)
    elif coloring in ["mcs", "mcs_type"]:
        mcs_list = []
        for org in orgs:
            for mcs_label, mcs_value in org.mcs.items():
                if mcs_label == mcs_label and mcs_value != {}:
                    mcs_list.append(mcs_value)
                else:
                    mcs_list.append(None)
        if coloring == "mcs":
            _apply_mcs_color(meshes, mcs_list)
        elif coloring == "mcs_type":
            _apply_mcs_type_color(meshes, mcs_list, types)
    elif coloring == "curvature":
        curvature_list = [org.morphology_map() for org in orgs]

        _apply_curvature_color(meshes, curvature_list)

    scene = trimesh.Scene(meshes)
    scene.export(str(file_path))

    return file_path, scene


def _apply_uniform_color(meshes, color=None):
    if color is None:
        color = (31, 119, 180, 255)

    for mesh in meshes:
        mesh.visual.vertex_colors[:] = color


def _apply_type_color(meshes, types):
    color_list = [
        (31, 119, 180, 255),
        (255, 127, 14, 255),
        (214, 39, 40, 255),
        (44, 160, 44, 255),
        (148, 103, 189, 255),
        (140, 86, 75, 255),
        (227, 119, 194, 255),
        (127, 127, 127, 255),
        (188, 189, 34, 255),
        (23, 190, 207, 255),
    ]
    color_map = dict(zip(set(types), color_list))

    for mesh, type in zip(meshes, types):
        mesh.visual.vertex_colors[:] = color_map[type]


def _apply_mcs_color(meshes, mcs):
    # first set uniform base color.
    _apply_uniform_color(meshes)

    # now apply color only to the mcs vertices
    for mesh, mcs_entries in zip(meshes, mcs):
        if mcs_entries is None:
            continue
        for mcs_target_label, mcs_data in mcs_entries.items():
            idx = mcs_data["vertices_index"]

            mesh.visual.vertex_colors[idx] = (0, 0, 255, 255)


def _apply_mcs_type_color(meshes, mcs, types):
    _apply_type_color(meshes, types)

    # now apply color only to the mcs vertices
    for mesh, mcs_entries in zip(meshes, mcs):
        if mcs_entries is None:
            continue
        for mcs_target_label, mcs_data in mcs_entries.items():
            idx = mcs_data["vertices_index"]

            mesh.visual.vertex_colors[idx] = (0, 0, 255, 255)


def _map_curvature_to_colors(curvature):
    # Normalize the integers
    min_val = min(curvature)
    max_val = max(curvature)
    normalized_curvature = [(i - min_val) / (max_val - min_val) for i in curvature]

    # Choose a colormap
    colormap = plt.cm.viridis

    # Map each normalized integer to a color
    colors = []
    for value in normalized_curvature:
        raw_color = colormap(value)
        color = [int(col * 255) for col in raw_color]
        colors.append(color)

    return colors


def _apply_curvature_color(meshes, curvatures):
    for mesh, curvature in zip(meshes, curvatures):
        colors = _map_curvature_to_colors(curvature)
        mesh.visual.vertex_colors[:] = colors


def _setup_blender(ply_filepath, scene):
    # Clear existing mesh, camera, and light
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import PLY file
    bpy.ops.wm.ply_import(filepath=str(ply_filepath))

    obj = bpy.context.active_object

    # Create a new material
    material = bpy.data.materials.new(name="VertexColorMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create a Vertex Color node
    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    vertex_color_node.layer_name = (
        "Col"  # This is the default name, change if your layer is named differently
    )

    # Create a Principled BSDF shader node
    shader_node = nodes.new(type="ShaderNodeBsdfPrincipled")

    # Create an Output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Link Vertex Color node to the Shader node's Base Color
    links.new(vertex_color_node.outputs["Color"], shader_node.inputs["Base Color"])

    # Link Shader node to the Output node
    links.new(shader_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the object
    if obj.data.materials:
        # Replace the existing material
        obj.data.materials[0] = material
    else:
        # Add the new material
        obj.data.materials.append(material)
    ###### finished coloring the mesh

    initial_cam_position = {
        "eye": {"x": 1, "y": 1, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    }

    center, radius = (
        scene.bounding_box.primitive.center,
        scene.bounding_sphere.primitive.radius,
    )

    # Create a light source
    bpy.ops.object.light_add(type="SUN", location=center + (2, 0, radius * 2))
    light = bpy.context.object
    light.data.energy = 5  # Adjust energy as needed
    light.name = "Sun"

    _update_cam(initial_cam_position, radius, center)


def _update_cam(camera_position, radius, center=None, lens_value=30):
    if isinstance(camera_position, dict):
        plotly_camera_eye = camera_position["eye"]
        # Convert Plotly parameters to Blender coordinates
        blender_camera_location_origin = (
            plotly_camera_eye["x"],
            plotly_camera_eye["y"],
            plotly_camera_eye["z"],
        )
    else:
        blender_camera_location_origin = camera_position

    if isinstance(center, dict):
        center = (center["x"], center["y"], center["z"])

    if "CameraFocus" in bpy.data.objects and center is None:
        camera_focus = bpy.data.objects["CameraFocus"]
        center = camera_focus.location

    elif "CameraFocus" not in bpy.data.objects and center is None:
        raise ValueError("Center is required when CameraFocus object is not present.")

    blender_camera_focus_raw = (center[0], center[2], center[1])

    blender_camera_focus = np.asarray(
        blender_camera_focus_raw
    )  # + plotly_relative_position

    # Calculate the direction vector from focus to origin location
    direction_vector = np.asarray(blender_camera_location_origin) - np.asarray(
        blender_camera_focus
    )
    # Normalize the direction vector
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
    # Set the camera location at the specified radius in the direction of the normalized vector
    blender_camera_location = (
        np.asarray(blender_camera_focus) + direction_vector_normalized * radius
    )
    light_position = (
        np.asarray(blender_camera_focus) + direction_vector_normalized * radius + 3
    )

    # Create or get the camera object
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=blender_camera_location)
        camera = (
            bpy.context.active_object
        )  # Corrected from bpy.context.object to bpy.context.active_object
    else:
        camera = bpy.data.objects["Camera"]
        camera.location = blender_camera_location

    # Create or get an empty object to act as the focus point
    if "CameraFocus" not in bpy.data.objects:
        bpy.ops.object.empty_add(location=blender_camera_focus)
        camera_focus = bpy.context.active_object
        camera_focus.name = "CameraFocus"

    else:
        camera_focus = bpy.data.objects["CameraFocus"]
        camera_focus.location = blender_camera_focus

    # Set camera to look at the focus point using a constraint
    if camera.constraints.get("Track To") is None:
        track_to_constraint = camera.constraints.new(type="TRACK_TO")
    else:
        track_to_constraint = camera.constraints["Track To"]
    track_to_constraint.target = camera_focus
    track_to_constraint.up_axis = "UP_Y"
    track_to_constraint.track_axis = "TRACK_NEGATIVE_Z"

    # Assuming 'camera' is your camera object from the script

    camera.data.lens = lens_value  # Focal length in mm
    camera.data.clip_end = np.inf
    bpy.context.scene.camera = bpy.data.objects["Camera"]

    # update light position
    light = bpy.data.objects["Sun"]
    light.location = light_position
    light.matrix_world.translation = light_position

    bpy.context.view_layer.update()
    return blender_camera_location, blender_camera_focus, lens_value


def _render_blender(
    output_path, resolution=(1800, 1800), engine="BLENDER_EEVEE", show_image=True
):
    # Adjust render settings
    bpy.context.scene.render.engine = engine  # Use 'BLENDER_EEVEE' or CYCLES
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # show camera object
    bpy.context.scene.camera = bpy.data.objects["Camera"]

    bpy.context.scene.render.filepath = str(output_path)

    bpy.ops.render.render(write_still=True)

    if show_image is True:
        # show the rendered image
        img = plt.imread(output_path)
        return plt.imshow(img)


def _live_camera_control(project, ids="*"):
    # Initialize the Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.camera_position = {
        "eye": {"x": 1, "y": 1, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    }

    # temp location to store preview image
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    # Plotly graph object for the 3D scatter plot
    fig = project.show(ids=ids)

    # Layout adjustments for the 3D plot
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=2),
            )
        ),
    )

    # find figure bounding sphere for later use
    vertices = np.empty((0, 3))
    for mesh in fig["data"]:
        # Convert mesh coordinates to NumPy arrays
        x_array = np.array(mesh["x"])
        y_array = np.array(mesh["y"])
        z_array = np.array(mesh["z"])

        # Stack x, y, z arrays into a single array of vertices
        mesh_vertices = np.stack((x_array, y_array, z_array), axis=-1)
        vertices = np.concatenate((vertices, mesh_vertices))

    center = np.mean(vertices, axis=0)

    # we don't need the vertices anymore
    vertices = None
    mesh_vertices = None

    # Dash layout with the Plotly graph and a div to display the camera position
    left_column = dbc.Col(
        html.Div(dcc.Graph(id="3d-graph", figure=fig), style={"width": "100%"}), width=7
    )

    right_column = dbc.Col(
        [
            dbc.Row(
                dbc.Button("Render Preview", id="update-render-camera"),
                style={"width": "30%", "margin-left": "10px"},
            ),
            dbc.Row(
                [
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Lens:", id="lens_label"),
                            dbc.Input(
                                id="lens_value",
                                type="number",
                                value=30,
                                style={"width": "30%"},
                            ),
                        ],
                    ),
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Distance:", id="distance_label"),
                            dbc.Input(
                                id="distance_value",
                                type="number",
                                value=10,
                                style={"width": "30%", "margin-down": "10px"},
                            ),
                        ],
                    ),
                ]
            ),
            dbc.Row(
                html.Img(
                    id="rendered-image", src="", style={"width": "100%"}
                )  # Initially empty Img tag
            ),
        ],
        width=5,
    )

    app.layout = dbc.Container(
        [
            html.H1("3D Scatter Plot with Camera Controls"),
            dbc.Row(
                [left_column, right_column]
            ),  # Organizes columns in a row with space around them
            html.Div(id="camera-position"),
        ],
        fluid=True,  # Sets the container to be fluid, using the full width of the viewport
    )

    @app.callback(
        Output("camera-position", "children"),
        Input("3d-graph", "relayoutData"),
        Input("lens_value", "value"),
        Input("distance_value", "value"),
    )
    def dash_update_blender_camera(relayoutData, lens_value, distance_value):
        if relayoutData is None:
            plotly_camera_eye = {"x": 1, "y": 1, "z": 1}
            plotly_camera_center = {"x": 0, "y": 0, "z": 0}

        else:
            # when changing to orbital camera mode the dict resets for some reason, so we just use the last entry
            if "scene.camera" in relayoutData:
                test_cam_position = relayoutData["scene.camera"]
                # Example Plotly camera parameters
                plotly_camera_eye = test_cam_position["eye"]
                plotly_camera_center = test_cam_position["center"]
                app.camera_position = {
                    "eye": plotly_camera_eye,
                    "center": plotly_camera_center,
                }

        _update_cam(app.camera_position, distance_value, center, lens_value)

    @app.callback(
        Output("rendered-image", "src", allow_duplicate=True),
        Input("update-render-camera", "n_clicks"),
        prevent_initial_call=True,
    )
    def _render_preview(n_clicks):
        # Create a temporary file to store the rendered image
        image_path = temp_file.name
        image_path = "blender_tests/preview.png"

        # Render the image using Blender and save it to the 'assets' directory
        project.render_blender(image_path, show_image=False)

        # Read the rendered image and encode it to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Format the base64-encoded string as a data URL
        data_url = f"data:image/png;base64,{encoded_image}"

        # Return the data URL as the source for the html.Img component
        return data_url

    # Run the Dash app

    def get_available_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # Bind to an available port chosen by the OS
            return s.getsockname()[1]  # Return the chosen port

    # Get an available port
    port = get_available_port()
    print(f"Running server on port {port}.")
    app.run_server(debug=True, port=port)
