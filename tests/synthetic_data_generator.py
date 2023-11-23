import numpy as np
import trimesh
import tempfile
import z5py
import pathlib
from skimage.measure import block_reduce
import json


def _array_trim(arr, margin=0):
    # small helper function to trim 3d arrays
    all = np.where(arr != 0)
    idx = ()
    for i in range(len(all)):
        idx += (np.s_[all[i].min() - margin : all[i].max() + margin + 1],)
    return arr[idx]


def generate_synthetic_dataset(
    n_objects=30, object_size=20, object_distance=100, seed=42, working_dir=None
):
    if working_dir is None:
        temp_dir = pathlib.Path(tempfile.mkdtemp())
    else:
        working_dir = pathlib.Path(working_dir)
        working_dir.mkdir(exist_ok=True)

        temp_dir = pathlib.Path(tempfile.mkdtemp(dir=working_dir))

    voxel_array, meshes = generate_synthetic_mesh_set(
        n_objects=n_objects,
        object_size=object_size,
        object_distance=object_distance,
        seed=seed,
    )

    resolution = [1, 1, 1]

    cebra_dir = temp_dir / "CebraEM"

    filename = cebra_dir / "images/bdv-n5" / "synth_data.n5"
    f = z5py.File(filename, use_zarr_format=False)
    group = f.create_group("setup0/timepoint0")

    downsampling_factors = ([1, 1, 1], [2, 2, 2], [4, 4, 4], [16, 16, 16])
    for i, factor in enumerate(downsampling_factors):
        down_sampled = block_reduce(voxel_array, factor[0], func=np.median)
        down_sampled = np.asarray(down_sampled, dtype="uint16")
        group.create_dataset(f"s{i}", data=down_sampled, dtype="uint16")

        # add downsampling factors to attributes.json
        attribute_json_filename = (
            filename / "setup0" / "timepoint0" / f"s{i}" / "attributes.json"
        )
        with open(attribute_json_filename, "r") as f:
            data = json.load(f)
        data["downsamplingFactors"] = factor
        with open(attribute_json_filename, "w") as f:
            json.dump(data, f)

    # create setup and timepoint attributes.json files
    setup0_dict = {"datatype": "uint16", "downsamplingFactors": downsampling_factors}
    with open(filename / "setup0" / "attributes.json", "w") as f:
        json.dump(setup0_dict, f)

    timepoint0_dict = {"multiscale": "true", "resolution": resolution}
    with open(filename / "setup0" / "timepoint0" / "attributes.json", "w") as f:
        json.dump(timepoint0_dict, f)

    # create xml file
    xml_str = _create_xml_file(voxel_array.shape, resolution)

    with open(cebra_dir / "images/bdv-n5" / "synth_data.xml", "wb") as f:
        f.write(xml_str.encode())

    # create dataset_json
    relativ_path = pathlib.Path("images/bdv-n5/synth_data.xml")
    dataset_dict = _create_dataset_json(str(relativ_path))
    with open(cebra_dir / "dataset.json", "w") as f:
        json.dump(dataset_dict, f)

    project_json_dict = {
        "datasets": ["CebraEM"],
        "defaultDataset": "CebraEM",
        "imageDataFormats": ["bdv.n5"],
        "specVersion": "0.2.0",
    }
    with open(temp_dir / "project.json", "w") as f:
        json.dump(project_json_dict, f)

    return temp_dir, meshes


def _create_dataset_json(relativ_path):
    dataset_dict = {
        "is2D": False,
        "sources": {
            "synth_data": {
                "image": {
                    "imageData": {
                        "bdv.n5": {
                            "relativePath": relativ_path,
                        }
                    }
                }
            }
        },
        "timepoints": 1,
        "views": {
            "default": {
                "isExclusive": False,
                "sourceDisplays": [
                    {
                        "imageDisplay": {
                            "color": "white",
                            "contrastLimits": [0, 255],
                            "name": "synth_data",
                            "opacity": 1.0,
                            "sources": ["synth_data"],
                        }
                    }
                ],
                "uiSelectionGroup": "bookmark",
            },
            "synth_data": {
                "isExclusive": False,
                "sourceDisplays": [
                    {
                        "imageDisplay": {
                            "color": "white",
                            "contrastLimits": [0, 255],
                            "name": "synth_data",
                            "opacity": 1.0,
                            "sources": ["synth_data"],
                        }
                    }
                ],
                "uiSelectionGroup": "images",
            },
        },
    }
    return dataset_dict


def _create_xml_file(size, resolution):
    xml_str = f"""
        <SpimData version="0.2">
        <BasePath type="relative">.</BasePath>
        <SequenceDescription>
            <ImageLoader format="bdv.n5">
            <n5 type="relative">synth_data.n5</n5>
            </ImageLoader>
            <ViewSetups>
            <Attributes name="channel">
                <Channel>
                <id>0</id>
                </Channel>
            </Attributes>
            <ViewSetup>
                <id>0</id>
                <name>synth_data</name>
                <size>{size[0]} {size[1]} {size[2]}</size>
                <voxelSize>
                <unit>micrometer</unit>
                <size>{resolution[0]} {resolution[1]} {resolution[2]}</size>
                </voxelSize>
                <attributes>
                <channel>0</channel>
                </attributes>
            </ViewSetup>
            </ViewSetups>
            <Timepoints type="range">
            <first>0</first>
            <last>0</last>
            </Timepoints>
        </SequenceDescription>
        <ViewRegistrations>
            <ViewRegistration timepoint="0" setup="0">
            <ViewTransform type="affine">
                <affine>{resolution[0]} 0.0 0.0 0.0 0.0 {resolution[1]} 0.0 0.0 0.0 0.0 {resolution[2]} 0.0</affine>
            </ViewTransform>
            </ViewRegistration>
        </ViewRegistrations>
        </SpimData>   """
    return xml_str


def generate_synthetic_mesh_set(
    n_objects=30, object_size=20, object_distance=100, seed=42
):
    np.random.seed(seed)

    # generation
    scales = (
        np.random.rand(n_objects, 3) * object_size + 10
    )  # Random scales between 10 and 30

    # translation

    # this main translation should remove negative numbers for the coordinates

    translations = np.random.randint(0, object_distance, (n_objects, 3))

    meshes = []
    meshes_to_remove = []
    collision_manager = trimesh.collision.CollisionManager()

    for i, (scale, translation) in enumerate(zip(scales, translations)):
        # Create an icosphere and scale it to create a random shape

        rand_object = np.random.rand()
        if rand_object < 0.7:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)
            mesh.apply_scale(scale)

        elif rand_object < 1:
            height = np.random.randint(1, scale, 1)[0]
            radius = np.random.randint(3, scale * 1.4, 1)[0]
            mesh = trimesh.creation.cylinder(radius=radius, height=height)

        # translate and rotate object, but don't make any overlaps

        mesh.apply_translation(translation)
        mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

        _, collision_partners = collision_manager.in_collision_single(
            mesh, return_names=True
        )
        if len(collision_partners) > 0:
            for collision_partner in collision_partners:
                mesh = mesh + meshes[collision_partner]["mesh"]

                # note which meshes to remove later
                meshes_to_remove.append(collision_partner)

                # Remove the collision partner from the collision manager
                collision_manager.remove_object(collision_partner)

        # Calculate volume and area
        volume = mesh.volume
        area = mesh.area
        center = mesh.centroid
        meshes.append(
            {
                "mesh": mesh,
                "scale": scale,
                "volume": volume,
                "area": area,
                "center": center,
            }
        )
        collision_manager.add_object(i, mesh)

    # Remove meshes that were merged
    for mesh_to_remove in sorted(meshes_to_remove, reverse=True):
        del meshes[mesh_to_remove]

    # voxelize meshes
    voxel_size = 1

    for mesh_dict in meshes:
        mesh = mesh_dict["mesh"]
        voxel = mesh.voxelized(voxel_size)

        voxel.fill()
        mesh_dict["voxelized"] = voxel

    # get min and max bounding box coordinates of all meshes
    min_bounding_box = np.array([np.inf, np.inf, np.inf])
    max_bounding_box = np.array([-np.inf, -np.inf, -np.inf])

    for mesh_dict in meshes:
        mesh = mesh_dict["mesh"]
        min_bounding_box = np.minimum(min_bounding_box, mesh.bounds[0])
        max_bounding_box = np.maximum(max_bounding_box, mesh.bounds[1])

    # set some arbitrary padding, maybe update this in the future

    voxel_array_size = 5 * object_distance * np.ones(3).astype(int)
    data_offset = 2 * object_distance * np.ones(3).astype(int)

    voxel_array = np.zeros(voxel_array_size)

    # Add the voxel data of each object to the voxel_array at the correct position
    for i, mesh_dict in enumerate(meshes):
        # Voxelizing the mesh
        voxelized = mesh_dict["voxelized"]

        position = mesh_dict["mesh"].bounds[0].astype(int) + data_offset
        position = position.astype(int)
        end_position = position + np.array(voxelized.matrix.shape)

        voxel_array[
            position[0] : end_position[0],
            position[1] : end_position[1],
            position[2] : end_position[2],
        ] += voxelized.matrix * (i + 1)

    voxel_array = _array_trim(voxel_array)
    voxel_array = voxel_array.astype(np.uint16)

    return voxel_array, meshes
