


#to-do:
#instead of manually setting camera angles, write a look-at function




import numpy as np
import matplotlib.pyplot as plt

Testmode = True

#-----    camera parameters     -----
pixel_width = 1000
pixel_height = 1000
fieldofview = 40    #in degrees
focal_length = 1.0  #arbitrary units, just a ratio

camera_pitch = np.deg2rad(50)    #rotation around x-axis
camera_roll = np.deg2rad(0)     #rotation around y-axis
camera_yaw = np.deg2rad(0)      #rotation around z-axis

x_offset = 0.0  #camera position offset in x
y_offset = 15.0  #camera position offset in y
z_offset = -12.0  #camera position offset in z

camera_position = np.array([x_offset, y_offset, z_offset])

#----- Parameters / units -----
M = .05          #BH mass in geometric units; Schwarzschild radius rs = 2M
rs = 2.0 * M
bh_pos = np.array([0.0, 0.0, 0.0])
r0 = camera_position - bh_pos  #vector from BH to camera position


def make_camera_rays(pixel_width, pixel_height, fieldofview, focal_length):

    aspect_ratio = pixel_width / pixel_height
    fov_rad = np.deg2rad(fieldofview)
    plane_height = 2 * focal_length * np.tan(fov_rad / 2)
    plane_width = plane_height * aspect_ratio


    #----- creating image plane -----
    xs = np.linspace(-plane_width/2, plane_width/2, pixel_width)
    ys = np.linspace(-plane_height/2, plane_height/2, pixel_height)
    px, py = np.meshgrid(xs, ys)
    pz = np.full_like(px, focal_length)


    #----- creating rays -----
    rays = np.stack((px,py,pz),axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays /= norms #normalize rays to represent direction

    #-----   camera rotation -----
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(camera_pitch), -np.sin(camera_pitch)],
                   [0, np.sin(camera_pitch),  np.cos(camera_pitch)]])
    
    Ry = np.array([[ np.cos(camera_roll), 0, np.sin(camera_roll)],
                   [0, 1, 0],
                   [-np.sin(camera_roll), 0, np.cos(camera_roll)]])
    
    Rz = np.array([[np.cos(camera_yaw), -np.sin(camera_yaw), 0],
                   [np.sin(camera_yaw),  np.cos(camera_yaw), 0],
                   [0, 0, 1]])

    # Apply yaw, then pitch, then roll
    R = Rz @ Rx @ Ry
    rays = rays @ R.T  # rotate each ray direction

    if Testmode == True:
        cy,cx = pixel_height//2, pixel_width//2
        print("center ray:", rays[cy, cx]) #expected: [0,0,1]

        top_center = rays[0,cx]
        theta = np.arccos(top_center[2]) #normalized to unit length, thus z-component is cos(theta)
        print("angle to top center ray:", np.rad2deg(theta)) #epected: 30 degrees   

        #-----   visualizing rays    -----
        sample_rate = 40
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,projection='3d')
        ax.quiver(
            np.zeros_like(px[::sample_rate, ::sample_rate]),
            np.zeros_like(py[::sample_rate, ::sample_rate]),
            np.zeros_like(pz[::sample_rate, ::sample_rate]),
            rays[::sample_rate, ::sample_rate, 0],
            rays[::sample_rate, ::sample_rate, 1],
            rays[::sample_rate, ::sample_rate, 2],
            length = 0.1,
            linewidths = .6,
            arrow_length_ratio = 0.07
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    return rays

rays = make_camera_rays(pixel_width=pixel_width, pixel_height=pixel_height, 
                        fieldofview=fieldofview, focal_length=focal_length)


#----- Impact parameter map b = || r0 x D || -----
cross_r0_D = np.cross(np.broadcast_to(r0, rays.shape), rays)  # (H, W, 3)
b = np.linalg.norm(cross_r0_D, axis=-1)                       # (H, W)

#----- Weak-field deflection Δ = 4M/b (radians) -----
eps = 1e-12
Delta = np.zeros_like(b)
mask_b = b > eps
Delta[mask_b] = 4.0 * M / b[mask_b]

#diagnostics for deflection math
if Testmode == True:
    Delta_deg = np.rad2deg(Delta)
    print("b stats:  min={:.4f}, median={:.4f}, max={:.4f}".format(b.min(), np.median(b), b.max()))
    print("Δ (deg):  min={:.4f}, median={:.4f}, max={:.4f}".format(Delta_deg[mask_b].min(),
                                                               np.median(Delta_deg[mask_b]),
                                                               Delta_deg[mask_b].max()))

    target_deg = 5.0
    M_suggest = (np.deg2rad(target_deg) * np.median(b[mask_b])) / 4.0
    print("Suggested M for ~{}° median deflection: {:.4f}".format(target_deg, M_suggest))

#----- Build rotation axis: n̂ ∝ D × (-r0) -----
axis = np.cross(rays, -np.broadcast_to(r0, rays.shape))   # (H, W, 3)
axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)

axis_unit = np.zeros_like(axis)
mask_axis = axis_norm[..., 0] > eps
axis_unit[mask_axis] = axis[mask_axis] / axis_norm[mask_axis]

# --- Rodrigues' rotation of D toward BH by angle Δ ---
cosD = np.cos(Delta)[..., None]  # (H,W,1)
sinD = np.sin(Delta)[..., None]

nxD = np.cross(axis_unit, rays)                      # n̂ × D
n_dot_D = np.sum(axis_unit * rays, axis=-1, keepdims=True)  # n̂·D

rotated = np.empty_like(rays)
rotated[mask_axis] = (
    rays[mask_axis] * cosD[mask_axis]
    + nxD[mask_axis] * sinD[mask_axis]
    + axis_unit[mask_axis] * n_dot_D[mask_axis] * (1.0 - cosD[mask_axis])
)
rotated[~mask_axis] = rays[~mask_axis]  # parallel case: no bend

# Re-normalize
rays_bent = rotated / np.linalg.norm(rotated, axis=-1, keepdims=True)





#create basic scene
zdisk = 0.0
inner_radius = 1.0
outer_radius = 4.0

t_vals = (zdisk-camera_position[2]) / rays_bent[...,2] #how much time for the ray to reach z disk
hit_x = camera_position[0] + t_vals * rays_bent[...,0] #x and y value when ray reaches z disk distance
hit_y = camera_position[1] + t_vals * rays_bent[...,1]

#solving for radius for each ray at zdisk
radii = np.sqrt(hit_x**2 + hit_y**2)

#where true, rays intersect with the disk
hit_mask = (t_vals > 0) & (radii >= inner_radius) & (radii <= outer_radius)

# Calculate polar coordinates for hits
angles = np.arctan2(hit_y, hit_x)  # range -pi to pi
angles_normalized = (angles + np.pi) / (2 * np.pi)  # 0 to 1

# Create checkerboard pattern
checker_u = np.floor(radii * 4)  # radial divisions (adjust 4 for more/less squares)
checker_v = np.floor(angles_normalized * 25)  # angular divisions (adjust 8 for more wedges)
checker_pattern = ((checker_u + checker_v) % 2).astype(float)

# Apply to image
image = np.zeros((pixel_height, pixel_width))
image[hit_mask] = checker_pattern[hit_mask]
plt.imshow(image, cmap = 'gray', origin = 'lower')
plt.gca().invert_yaxis()
plt.show()