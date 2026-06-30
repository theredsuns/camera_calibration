from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    
    camera_frame = LaunchConfiguration('camera_frame')
    tag_frame = LaunchConfiguration('tag_frame')
    target_tag_id = LaunchConfiguration('target_tag_id')
    tag_size = LaunchConfiguration('tag_size')
    use_direct_camera = LaunchConfiguration('use_direct_camera')
    camera_device_id = LaunchConfiguration('camera_device_id')
    camera_fx = LaunchConfiguration('camera_fx')
    camera_fy = LaunchConfiguration('camera_fy')
    camera_cx = LaunchConfiguration('camera_cx')
    camera_cy = LaunchConfiguration('camera_cy')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    use_stereo = LaunchConfiguration('use_stereo')
    publish_tf = LaunchConfiguration('publish_tf')
    publish_marker = LaunchConfiguration('publish_marker')
    enable_compensation = LaunchConfiguration('enable_compensation')
    print_precise_pose = LaunchConfiguration('print_precise_pose')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_frame',
            description='Camera frame ID'
        ),
        DeclareLaunchArgument(
            'tag_frame',
            default_value='tag_36h11_id2',
            description='Tag frame ID'
        ),
        DeclareLaunchArgument(
            'target_tag_id',
            default_value='2',
            description='Target AprilTag ID (36h11 family)'
        ),
        DeclareLaunchArgument(
            'tag_size',
            default_value='0.06',
            description='Tag size in meters (6cm)'
        ),
        DeclareLaunchArgument(
            'use_direct_camera',
            default_value='true',
            description='Use direct camera capture instead of ROS topics'
        ),
        DeclareLaunchArgument(
            'camera_device_id',
            default_value='0',
            description='Camera device ID (0 for /dev/video0)'
        ),
        DeclareLaunchArgument(
            'camera_fx',
            default_value='500.0',
            description='Camera focal length X'
        ),
        DeclareLaunchArgument(
            'camera_fy',
            default_value='500.0',
            description='Camera focal length Y'
        ),
        DeclareLaunchArgument(
            'camera_cx',
            default_value='320.0',
            description='Camera principal point X'
        ),
        DeclareLaunchArgument(
            'camera_cy',
            default_value='240.0',
            description='Camera principal point Y'
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='640',
            description='Image width'
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='480',
            description='Image height'
        ),
        DeclareLaunchArgument(
            'use_stereo',
            default_value='false',
            description='Use stereo ZED camera (when use_direct_camera=false)'
        ),
        DeclareLaunchArgument(
            'publish_tf',
            default_value='true',
            description='Publish TF transform'
        ),
        DeclareLaunchArgument(
            'publish_marker',
            default_value='true',
            description='Publish visualization marker'
        ),
        DeclareLaunchArgument(
            'enable_compensation',
            default_value='true',
            description='Enable skew and edge error compensation'
        ),
        DeclareLaunchArgument(
            'print_precise_pose',
            default_value='true',
            description='Print precise pose with 3 decimal places'
        ),
        
        Node(
            package='apriltag_zed_visp',
            executable='apriltag_detector',
            name='apriltag_detector',
            output='screen',
            parameters=[{
                'camera_frame': camera_frame,
                'tag_frame': tag_frame,
                'target_tag_id': target_tag_id,
                'tag_size': tag_size,
                'use_direct_camera': use_direct_camera,
                'camera_device_id': camera_device_id,
                'camera_fx': camera_fx,
                'camera_fy': camera_fy,
                'camera_cx': camera_cx,
                'camera_cy': camera_cy,
                'image_width': image_width,
                'image_height': image_height,
                'use_stereo': use_stereo,
                'publish_tf': publish_tf,
                'publish_marker': publish_marker,
                'enable_compensation': enable_compensation,
                'print_precise_pose': print_precise_pose,
            }],
            remappings=[
                ('zed/left/image_rect_color', '/zed/left/image_rect_color'),
                ('zed/right/image_rect_color', '/zed/right/image_rect_color'),
                ('zed/left/camera_info', '/zed/left/camera_info'),
            ]
        )
    ])
