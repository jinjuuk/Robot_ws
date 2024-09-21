from setuptools import find_packages, setup

package_name = 'test_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jinjuuk',
    maintainer_email='jinjuuk2@gmail.com',
    description='TODO: yolov5_node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_node = test_pkg.test_node:main',
            'local_compressed_receive_node = test_pkg.local_compressed_receive:main',
            'initial_test_camera_node = test_pkg.initial_test_camera:main',
            'object_tracking_node = test_pkg.Object_tracking_node:main',
            'all_in_one_node = test_pkg.all_in_one_node:main',
            'aio_test_node = test_pkg.aio_test_node:main',
            'visual_test_node = test_pkg.visual_test_node:main',
            'udp_test_node = test_pkg.udp_test_node:main',
            'udp_visual_test_node = test_pkg.udp_visual_test:main',
            'test_all_in_one_visual_node = test_pkg.all_in_one_visual_test_node:main',
            'obj_tracking_lighting_node = test_pkg.obj_tracking_lightening:main',
            'AllInOneNode_Up_node = test_pkg.ot_disp_up:main',
            'ALLInOneNode_Down_node = test_pkg.ot_disp_down:main',
            'ot_com_test_node = test_pkg.ot_com_test:main',
            'all_in_one_multi_sim_node = test_pkg.all_in_one_multi_sim_node:main',
            'mr_do_node = test_pkg.MR_DO:main',
            'ot_disp_down_node = test_pkg.ot_disp_down:main',
            'ot_disp_up_node = test_pkg.ot_disp_up:main',

        ],
    },
)
