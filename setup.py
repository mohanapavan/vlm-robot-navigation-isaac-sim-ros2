from glob import glob

from setuptools import find_packages, setup

package_name = 'vlm_nav'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vlm_nav maintainers',
    maintainer_email='maintainers@example.com',
    description='Language-guided semantic navigation with Isaac Sim, ROS 2, Nav2, GroundingDINO and Qwen2.5-VL.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'build_scene_graph = vlm_nav.build_scene_graph:main',
            'robot_brain = vlm_nav.robot_brain:main',
            'map_relay = vlm_nav.map_relay:main',
            'record_bag = vlm_nav.record_bag:main',
        ],
    },
)
