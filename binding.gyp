{
    'variables': {
        'lite_dir%': '', # Path of Paddle Lite prebuild libraries for Windows x86
    },
    "targets": [
        {
            'target_name': "paddlenode",
            'sources': ["nodehelper.h","paddlelib.h","paddlelib.cc","paddlenode.cc"],
            'defines': [
            ],
            'include_dirs': [
                "<(lite_dir)/cxx/include",
                "<(lite_dir)/third_party/mklml/include"
            ],
            'libraries': [
                "-l<(lite_dir)/cxx/lib/libpaddle_api_light_bundled.lib",
                "-l<(lite_dir)/third_party/mklml/lib/libiomp5md.lib",
                "-l<(lite_dir)/third_party/mklml/lib/mklml.lib",
                "-lshlwapi.lib"
            ]
        }
    ]
}
