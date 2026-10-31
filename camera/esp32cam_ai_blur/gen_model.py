# PlatformIO pre-build script: embed the ESP-DL pedestrian model.
#
# The pedestrian_detect component packs its model and turns it into an assembler file with
# CMake custom commands, which PlatformIO does not run - the build then fails with
# "pedestrian_detect.espdl.S not found". This script runs the same two steps with
# Espressif's own tools before the build.
import os
import subprocess
import sys

Import("env")  # noqa: F821  (provided by PlatformIO)

project = env.subst("$PROJECT_DIR")
build = env.subst("$BUILD_DIR")
espdl = os.path.join(project, "managed_components", "espressif__esp-dl")
model = os.path.join(project, "managed_components", "espressif__pedestrian_detect",
                     "models", "s3", "pedestrian_detect_pico_s8_v1.espdl")

if not os.path.isfile(model):
    # first build: components are downloaded during it; run the build a second time
    print("gen_model: model not downloaded yet - build again after this run")
else:
    packed = os.path.join(build, "espdl_models", "pedestrian_detect.espdl")
    source = os.path.join(build, "pedestrian_detect.espdl.S")
    os.makedirs(os.path.dirname(packed), exist_ok=True)
    if not os.path.isfile(source) or os.path.getmtime(source) < os.path.getmtime(model):
        subprocess.check_call([sys.executable, os.path.join(espdl, "fbs_loader", "pack_espdl_models.py"),
                               "--model_path", model, "--out_file", packed])
        cmake = os.path.join(env.PioPlatform().get_package_dir("tool-cmake"), "bin", "cmake")
        subprocess.check_call([cmake, "-D", f"DATA_FILE={packed}", "-D", f"SOURCE_FILE={source}",
                               "-D", "FILE_TYPE=BINARY", "-P",
                               os.path.join(espdl, "fbs_loader", "cmake", "data_file_embed_asm_aligned.cmake")],
                              cwd=build)
        print("gen_model: embedded", os.path.basename(model))
