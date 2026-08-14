import numpy as np
import plotly.graph_objects as go

def create_camera_wireframe(R, t, K, width, height, scale=0.2):
    """
    Create camera wireframe visualization.
    
    Returns vertices and edges for camera pyramid.
    """
    # Camera center in world coordinates
    C = -R @ t
    
    # Define image corners in pixel coordinates
    corners_2d = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T
    
    # Unproject to camera coordinates at depth=scale
    K_inv = np.linalg.inv(K)
    corners_cam = K_inv @ corners_2d * scale
    
    # Transform to world coordinates
    corners_world = R @ corners_cam + C.reshape(3, 1)
    
    # Camera pyramid vertices (center + 4 corners)
    vertices = np.hstack([C.reshape(3, 1), corners_world]).T
    
    # Edges (connecting center to corners and corners to each other)
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Corner to corner
    ]
    
    return vertices, edges


def visualize_all_cameras(V, F, Rs, ts, K, width, height, output_file, P_points=None):
    fig = go.Figure()
    
    fig.add_trace(go.Mesh3d(
        x=V[:, 0],
        y=V[:, 1],
        z=V[:, 2],
        i=F[:, 0],
        j=F[:, 1],
        k=F[:, 2],
        color='lightblue',
        opacity=0.3,
        name='Original Mesh',
        showscale=False
    ))

    if P_points is not None:
        fig.add_trace(go.Scatter3d(
            x=P_points[:, 0],
            y=P_points[:, 1],
            z=P_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='royalblue'
            ),
            name='Points'
        ))

    for i, (R, t) in enumerate(zip(Rs, ts)):
        cam_vertices, cam_edges = create_camera_wireframe(R, t, K, width, height, scale=0.25)
        cam_lines = []
        for edge in cam_edges:
            cam_lines.extend([cam_vertices[edge[0]], cam_vertices[edge[1]], [None, None, None]])
        if cam_lines:
            cam_lines = np.array(cam_lines[:-1])
            fig.add_trace(go.Scatter3d(
                x=cam_lines[:, 0],
                y=cam_lines[:, 1],
                z=cam_lines[:, 2],
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=2, color='green'),
                name=f'Camera {i}'
            ))
        # Camera center
        C = -R @ t
        fig.add_trace(go.Scatter3d(
            x=[C[0]],
            y=[C[1]],
            z=[C[2]],
            mode='markers',
            marker=dict(size=2, color='darkgreen', symbol='diamond'),
            name=f'Camera {i} Center'
        ))

    fig.update_layout(
        title='All Cameras Visualization',
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
    )

    fig.write_html(output_file)
    print(f"Saved all cameras visualization to {output_file}")



def visualize_scan_selection(title, V, F, P_points, P_scan, N_scan, Rs_prev, ts_prev, R, t, K, width, height, output_file):
    """
    Create interactive Plotly visualization.
    """
    # Create figure
    fig = go.Figure()
    
    # 1. Add mesh
    fig.add_trace(go.Mesh3d(
        x=V[:, 0],
        y=V[:, 1],
        z=V[:, 2],
        i=F[:, 0],
        j=F[:, 1],
        k=F[:, 2],
        color='lightblue',
        opacity=0.3,
        name='Original Mesh',
        showscale=False
    ))
    
    # 2. Add point cloud
    if P_points is not None:
        fig.add_trace(go.Scatter3d(
            x=P_points[:, 0],
            y=P_points[:, 1],
            z=P_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='royalblue'
            ),
            name='Previous Points'
        ))
    
    fig.add_trace(go.Scatter3d(
        x=P_scan[:, 0],
        y=P_scan[:, 1],
        z=P_scan[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='orchid'
        ),
        name='Scanned Points'
    ))
    
    # 3. Add normals as small lines
    # Sample normals for visualization (showing all might be too dense)
    num_normals_to_show = min(1000, len(P_scan))
    sample_indices = np.random.choice(len(P_scan), num_normals_to_show, replace=False)
    
    normal_scale = 0.05  # Scale factor for normal visualization
    normal_lines = []
    for idx in sample_indices:
        start = P_scan[idx]
        end = start + N_scan[idx] * normal_scale
        normal_lines.extend([start, end, [None, None, None]])
    
    if normal_lines:
        normal_lines = np.array(normal_lines[:-1])  # Remove last None
        fig.add_trace(go.Scatter3d(
            x=normal_lines[:, 0],
            y=normal_lines[:, 1],
            z=normal_lines[:, 2],
            mode='lines',
            line=dict(color='red', width=1),
            name='Scanned Normals',
            showlegend=True
        ))
    
    # 4. Add camera wireframe
    cam_vertices, cam_edges = create_camera_wireframe(R, t, K, width, height, scale=0.25)
    
    # Camera pyramid lines
    cam_lines = []
    for edge in cam_edges:
        cam_lines.extend([cam_vertices[edge[0]], cam_vertices[edge[1]], [None, None, None]])
    
    if cam_lines:
        cam_lines = np.array(cam_lines[:-1])
        fig.add_trace(go.Scatter3d(
            x=cam_lines[:, 0],
            y=cam_lines[:, 1],
            z=cam_lines[:, 2],
            mode='lines+markers',
            line=dict(color='orchid', width=3),
            marker=dict(size=2, color='orchid'),
            name='Camera'
        ))
    
    # Add camera center as a distinct point
    C = -R @ t
    fig.add_trace(go.Scatter3d(
        x=[C[0]],
        y=[C[1]],
        z=[C[2]],
        mode='markers',
        marker=dict(size=2, color='orchid', symbol='diamond'),
        name='Camera Center'
    ))
    
    if Rs_prev is not None and ts_prev is not None:
        cam_lines = []
        for R_prev, t_prev in zip(Rs_prev, ts_prev):
            cam_vertices, cam_edges = create_camera_wireframe(R_prev, t_prev, K, width, height, scale=0.25)
            for edge in cam_edges:
                cam_lines.extend([cam_vertices[edge[0]], cam_vertices[edge[1]], [None, None, None]])
        if cam_lines:
            cam_lines = np.array(cam_lines[:-1])
            fig.add_trace(go.Scatter3d(
                x=cam_lines[:, 0],
                y=cam_lines[:, 1],
                z=cam_lines[:, 2],
                mode='lines+markers',
                line=dict(color='royalblue', width=3),
                marker=dict(size=2, color='royalblue'),
                name='Previous Cameras'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        width=1200,
        height=800
    )
    
    # Save HTML
    fig.write_html(output_file)
    print(f"Saved visualization to {output_file}")