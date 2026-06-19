# add_morphs.py
# Adds viseme + facial expression morph targets directly to GLB file
# Run: python D:\vra\add_morphs.py

from pygltflib import GLTF2
import json, struct, base64, copy, numpy as np

GLB_IN  = r"D:\vra\avatar\assets\model.glb"
GLB_OUT = r"D:\vra\avatar\assets\avatar_morph.glb"

MORPH_NAMES = [
    'viseme_PP','viseme_FF','viseme_TH','viseme_DD',
    'viseme_kk','viseme_CH','viseme_SS','viseme_nn',
    'viseme_RR','viseme_aa','viseme_E','viseme_I',
    'viseme_O','viseme_U','mouthOpen','mouthSmile',
    'jawOpen','eyeBlinkLeft','eyeBlinkRight','browInnerUp',
]

print("Loading GLB...")
gltf = GLTF2().load(GLB_IN)

# Get binary buffer
binary = gltf.binary_blob()
if binary is None:
    # Load from buffer uri
    import os
    buf = gltf.buffers[0]
    buf_path = os.path.join(os.path.dirname(GLB_IN), buf.uri)
    with open(buf_path, 'rb') as f:
        binary = f.read()

print(f"Meshes found: {len(gltf.meshes)}")

for mesh_idx, mesh in enumerate(gltf.meshes):
    print(f"\nProcessing mesh: {mesh.name}")
    
    for prim in mesh.primitives:
        # Get POSITION accessor to know vertex count
        pos_acc_idx = prim.attributes.POSITION
        pos_acc = gltf.accessors[pos_acc_idx]
        vertex_count = pos_acc.count
        print(f"  Vertices: {vertex_count}")

        # Create zero-displacement morph target for each morph name
        # Each morph target needs a POSITION accessor with all zeros
        zero_data = np.zeros((vertex_count, 3), dtype=np.float32)
        zero_bytes = zero_data.tobytes()

        if prim.targets is None:
            prim.targets = []

        existing_count = len(prim.targets)
        print(f"  Existing morph targets: {existing_count}")

        for morph_name in MORPH_NAMES:
            # Check if already exists in mesh extras
            if mesh.extras is None:
                mesh.extras = {}
            if 'targetNames' not in mesh.extras:
                mesh.extras['targetNames'] = []
            
            if morph_name in mesh.extras['targetNames']:
                print(f"  ⏭ Skipping (exists): {morph_name}")
                continue

            # Add zero bytes to binary buffer
            current_buf_len = len(binary)
            binary += zero_bytes

            # Pad to 4-byte alignment
            pad = (4 - len(binary) % 4) % 4
            binary += b'\x00' * pad

            # Create buffer view
            from pygltflib import BufferView
            bv = BufferView()
            bv.buffer = 0
            bv.byteOffset = current_buf_len
            bv.byteLength = len(zero_bytes)
            gltf.bufferViews.append(bv)
            bv_idx = len(gltf.bufferViews) - 1

            # Compute min/max for accessor
            from pygltflib import Accessor
            acc = Accessor()
            acc.bufferView   = bv_idx
            acc.byteOffset   = 0
            acc.componentType = 5126  # FLOAT
            acc.count        = vertex_count
            acc.type         = "VEC3"
            acc.min          = [0.0, 0.0, 0.0]
            acc.max          = [0.0, 0.0, 0.0]
            gltf.accessors.append(acc)
            acc_idx = len(gltf.accessors) - 1

            # Add morph target
            prim.targets.append({"POSITION": acc_idx})
            mesh.extras['targetNames'].append(morph_name)
            print(f"  ✅ Added: {morph_name}")

# Update buffer length
gltf.buffers[0].byteLength = len(binary)

# Save
gltf.set_binary_blob(binary)
gltf.save(GLB_OUT)

print(f"\n✅ Saved to: {GLB_OUT}")
print(f"   File size: {len(binary) / 1024 / 1024:.2f} MB")