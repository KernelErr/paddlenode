{
    'variables': {
        'lite_dir%': '', #C:/inference_lite_lib.win.x86.MSVC.C++_static.py37.full_publish
        'opencv_dir%': '' #C:/opencv/build
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
                "<(lite_dir)/cxx/include",
                "<(lite_dir)/third_party/mklml/include",
                "<(opencv_dir)/include"
            ],
            'libraries': [
                "-l<(lite_dir)/cxx/lib/libpaddle_api_light_bundled.lib",
                "-l<(lite_dir)/third_party/mklml/lib/libiomp5md.lib",
                "-l<(lite_dir)/third_party/mklml/lib/mklml.lib",
                "-lshlwapi.lib",
                "-l<(opencv_dir)/x64/vc15/lib/opencv_world440.lib",
                "-l<(opencv_dir)/x64/vc15/lib/opencv_world440d.lib"
            ]
        }
    ]
}
