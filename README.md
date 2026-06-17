# CP2PBR

-----> [Download the Blender plugin from here!](https://github.com/Riccardo-Foschi/CP2PBR/releases/download/v1.7/CP2PBR1-7.zip)<-----

CM2PBR is a Blender plugin that generates Metalness, Roughness, Albedo, and Normal maps by post-processing cross-polarized and non-cross-polarized textures. The two input textures must share the same UV layout.

The workflow starts from a standard photogrammetry pipeline using two image sets:
- one captured with cross-polarization filters on both the lights and the camera lens;
- one captured without cross-polarization.

Two meshes are reconstructed from the two sets of images and automatically aligned with a photogrammetric software of choice. The non-polarized texture is then projected onto the mesh generated from the cross-polarized images. This mesh is usually preferred because it tends to provide higher geometric accuracy, since removing glossy reflections improves feature matching during photogrammetry.

This texture reprojection step ensures that both textures share matching UVs, which is essential for CM2PBR.

The CP2PBR plugin requires manual tuning of several parameters to extract the different PBR maps, and physically plausible results are not guaranteed in every case. For example, objects with highly reflective non-metallic glossy surfaces and metallic patches may be difficult to process correctly, since glossy patches' brightness can exceed that of metallic areas which might prevent the user to be able to isolate them correctly.



<img width="582" height="1152" alt="CP2PBR_UI" src="https://github.com/user-attachments/assets/ac3ad6ce-87ed-4e66-9d5e-a999dc29587c" />
