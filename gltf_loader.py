from collections import namedtuple
import numpy as np
from pathlib import Path
import pygltflib


Model = namedtuple('Model', 'name nodes ordered_node_indexes meshes animations skins')
Mesh = namedtuple('Mesh', 'name primitives')
Primitive = namedtuple('Primitive', 'name material triangles vertices normals uvs joints weights')
AnimationSampler = namedtuple('AnimationSampler', 'interpolation keyframe_times keyframe_values')
AnimationChannel = namedtuple('AnimationChannel', 'sampler node path')
Animation = namedtuple('Animation', 'samplers channels, duration')
Skin = namedtuple('Skin', 'joints inverse_bind_matrices')


def get_dtype_cnt(accessor):
    dtype = None
    if accessor.componentType == pygltflib.BYTE:
        dtype = np.int8
    elif accessor.componentType == pygltflib.UNSIGNED_BYTE:
        dtype = np.uint8
    elif accessor.componentType == pygltflib.SHORT:
        dtype = np.int16
    elif accessor.componentType == pygltflib.UNSIGNED_SHORT:
        dtype = np.uint16
    elif accessor.componentType == pygltflib.UNSIGNED_INT:
        dtype = np.uint32
    elif accessor.componentType == pygltflib.FLOAT:
        dtype = np.float32

    cnt = 0
    if accessor.type == 'VEC4':
        cnt = 4
    elif accessor.type == 'VEC3':
        cnt = 3
    elif accessor.type == 'VEC2':
        cnt = 2
    elif accessor.type == 'SCALAR':
        cnt = 1

    return dtype, cnt
        

def load_accessor_data(gltf, accessor):
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    dtype, cnt = get_dtype_cnt(accessor)
    clipped_data = data[buffer_view.byteOffset : buffer_view.byteOffset + buffer_view.byteLength]
    elements = np.frombuffer(clipped_data, dtype=dtype)
    if cnt > 1:
        elements = np.reshape(elements, (-1, cnt))
    return elements


def load_model(fname):
    gltf = pygltflib.GLTF2().load(fname)

    meshes = []
    for mesh in gltf.meshes:
        primitives = []
        for primitive in mesh.primitives:
            triangles = load_accessor_data(gltf, gltf.accessors[primitive.indices])
            vertices = load_accessor_data(gltf, gltf.accessors[primitive.attributes.POSITION])
            normals = load_accessor_data(gltf, gltf.accessors[primitive.attributes.NORMAL])
            uvs = load_accessor_data(gltf, gltf.accessors[primitive.attributes.TEXCOORD_0])
            joints = weights = None
            if primitive.attributes.JOINTS_0:
                joints = load_accessor_data(gltf, gltf.accessors[primitive.attributes.JOINTS_0])
                weights = load_accessor_data(gltf, gltf.accessors[primitive.attributes.WEIGHTS_0])
            assert vertices.dtype == np.float32
            assert normals.dtype == np.float32
            assert uvs.dtype == np.float32
            assert len(vertices) == len(normals) == len(uvs)
            if joints is not None:
                assert joints.dtype in (np.uint8, np.uint16, np.uint32)
                assert weights.dtype == np.float32
                assert len(vertices) == len(joints) == len(weights)
                joints = np.array(joints, dtype=np.uint16)
            assert np.max(triangles) == len(vertices) - 1
            primitives.append(Primitive(mesh.name, primitive.material, triangles, vertices, normals, uvs, joints, weights))
        meshes.append(Mesh(mesh.name, primitives))

    animations = []
    for anim in gltf.animations:
        samplers = []
        channels = []
        duration = 0
        for sampler in anim.samplers:
            keyframe_times = load_accessor_data(gltf, gltf.accessors[sampler.input])
            keyframe_values = load_accessor_data(gltf, gltf.accessors[sampler.output])
            s = AnimationSampler(sampler.interpolation, keyframe_times, keyframe_values)
            samplers.append(s)
            if keyframe_times[-1] > duration:
                duration = keyframe_times[-1]

        for chnl in anim.channels:
            c = AnimationChannel(chnl.sampler, chnl.target.node, chnl.target.path)
            channels.append(c)

        a = Animation(samplers, channels, duration)
        animations.append(a)

        skins = []
        for skin in gltf.skins:
            joints = skin.joints
            inverse_bind_matrices = load_accessor_data(gltf, gltf.accessors[skin.inverseBindMatrices])
            inverse_bind_matrices = inverse_bind_matrices.reshape((-1, 16))
            assert inverse_bind_matrices.dtype == np.float32
            skins.append(Skin(joints, inverse_bind_matrices))

    ordered_node_indexes = order_nodes_root_first(gltf.nodes)

    root = gltf.nodes[ordered_node_indexes[0]]
    root.rotation = np.array([0,0,-0.707,0.707], dtype=np.float32) # rotate root node -90 deg around Z axis

    name = Path(fname).stem

    return Model(name, gltf.nodes, ordered_node_indexes, meshes, animations, skins)


def order_nodes_root_first(nodes):
    '''
        Returns the nodes sorted so that the parents come first. This helps make transforming bone chain hierarchies trivial.
    '''
    parent_indexes = [-1] * len(nodes)
    for node in nodes:
        node.parent_index = -1
    for i, node in enumerate(nodes):
        for child_node_index in node.children:
            nodes[child_node_index].parent_index = i
    ordered_parent_indexes = {}
    for i in range(len(nodes)):
        def add_node(j):
            if j in ordered_parent_indexes:
                return
            parent_index = nodes[j].parent_index
            if parent_index >= 0:
                add_node(parent_index)
            ordered_parent_indexes[j] = 1
        add_node(i)
    return list(ordered_parent_indexes)
