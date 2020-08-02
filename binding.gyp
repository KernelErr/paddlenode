{
    'variables': {
        'paddle_lib%': '', #C:/fluid_inference_install_dir
        'opencv_dir%': ''  # C:/opencv/build
    },
    "targets": [
        {
            'target_name': "paddlenode",
            'sources': [
                "./src/nodehelper.h",
                "./src/paddlelib.cc",
                "./src/paddlelib.h",
                "./src/paddlenode.cc",
            ],
            'include_dirs': [
                "<(paddle_lib)",
                "<(paddle_lib)/third_party/install/glog/include",
                "<(paddle_lib)/third_party/install/gflags/include",
                "<(paddle_lib)/third_party/install/protobuf/include",
                "<(paddle_lib)/third_party/install/xxhash/include",
                "<(paddle_lib)/third_party/install/mkldnn/include",
                "<(paddle_lib)/third_party/install/mklml/include",
                "<(opencv_dir)/include"
            ],
            'libraries': [
                "-l<(paddle_lib)/third_party/install/glog/lib/glog.lib",
                "-l<(paddle_lib)/third_party/install/gflags/lib/gflags_static.lib",
                "-l<(paddle_lib)/third_party/install/protobuf/lib/libprotobuf.lib",
                "-l<(paddle_lib)/third_party/install/xxhash/lib/xxhash.lib",
                "-l<(paddle_lib)/third_party/install/mkldnn/lib/mkldnn.lib",
                "-l<(paddle_lib)/third_party/install/mklml/lib/mklml.lib",
                "-l<(paddle_lib)/third_party/install/mklml/lib/libiomp5md.lib",
                "-l<(paddle_lib)/paddle/lib/libpaddle_fluid.lib",
                "-lshlwapi.lib",
                "-l<(opencv_dir)/x64/vc15/lib/opencv_world440.lib",
                "-l<(opencv_dir)/x64/vc15/lib/opencv_world440d.lib"
            ]
        }
    ]
}
