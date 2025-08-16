


#to-do:
#update to use full GR DEs on rays within a certain radius of the BH
#add coloration to accretion disk
#make slideshow function to animate change in theta angle of camera

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import io


Testmode = False    #test mode toggles extra print statements for debugging/sanity checks, ray visualization 
Checkering = True   #creates a checkerboard pattern on the disk to better visualize distortion
ShowImage = False   #if True, will show the final image after ray tracing
Gifmaker = True    #if True, will create a gif of the camera orienting from above to below the black hole

#----- camera parameters -----
pixel_width = 1000
pixel_height = 1000
fieldofview = 40    #in degrees
focal_length = 1.0  #arbitrary units, just a ratio

# Spherical camera positioning (relative to black hole at bh_pos)
camera_radius = 18.0  # distance from black hole
camera_theta = np.deg2rad(75)  # polar angle from +z axis (0 = above, 90 = in x-y plane)
camera_phi = np.deg2rad(0)     # azimuthal angle from +x axis in x-y plane

#create basic accretion disk scene
bh_pos = np.array([0.0, 0.0, 0.0]) #this will likely always be the origin
zdisk = 0.0                        #z offset for disk only; should always match bh z pos
inner_radius = 1.0
outer_radius = 4.0


#----- Parameters / units -----
M = .05          #BH mass in geometric units; Schwarzschild radius rs = 2M
rs = 2.0 * M


#----- GIF creation parameters -----
frames = []
framecount = 60
theta_degree_sweep = 30
phi_degree_sweep = 360
theta_rad_sweep = np.deg2rad(theta_degree_sweep)
phi_rad_sweep = np.deg2rad(phi_degree_sweep)

def sph_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

#camera position in Cartesian coordinates
camera_cart = sph_to_cart(camera_radius, camera_theta,camera_phi)
x_cam = bh_pos[0] + camera_cart[0]
y_cam = bh_pos[1] + camera_cart[1]
z_cam = bh_pos[2] + camera_cart[2]
camera_position = np.array([x_cam, y_cam, z_cam]) #relative to black hole position
  #vector from BH to camera position




def make_camera_rays(pixel_width, pixel_height, fieldofview, focal_length, camera_position):

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

    #-----   camera look-at rotation -----
    # Forward vector (from camera to black hole)
    forward = (bh_pos - camera_position)
    forward /= np.linalg.norm(forward)

    # Choose up vector (z up, unless forward is parallel to z)
    up_guess = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(forward, up_guess)), 1.0):
        up_guess = np.array([0, 1, 0])
    right = np.cross(up_guess, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Rotation matrix: columns are right, up, forward
    R = np.stack([right, up, forward], axis=1)
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


def create_image(rays):
    
    #----- Impact parameter map b = || r0 x D || -----
    r0 = camera_position - bh_pos
    cross_r0_D = np.cross(np.broadcast_to(r0, rays.shape), rays)  # (H, W, 3)
    b = np.linalg.norm(cross_r0_D, axis=-1)                       # (H, W)
    
    #----- Weak-field deflection Δ = 4M/b (radians) -----
    #approximation of gravitational deflection accurate in weak G fields
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

    #----- Rodrigues' rotation of D toward BH by angle Δ -----
    cosD = np.cos(Delta)[..., None]  # (H,W,1)
    sinD = np.sin(Delta)[..., None]

    nxD = np.cross(axis_unit, rays)                             # n̂ × D
    n_dot_D = np.sum(axis_unit * rays, axis=-1, keepdims=True)  # n̂·D

    rotated = np.empty_like(rays)
    rotated[mask_axis] = (rays[mask_axis] * cosD[mask_axis]
                        + nxD[mask_axis] * sinD[mask_axis]
                        + axis_unit[mask_axis] * n_dot_D[mask_axis] * (1.0 - cosD[mask_axis]))
    
    rotated[~mask_axis] = rays[~mask_axis]  # parallel case: no bend

    # Re-normalize
    rays_bent = rotated / np.linalg.norm(rotated, axis=-1, keepdims=True)


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
    checker_u = np.floor(radii * 5)  # radial divisions
    checker_v = np.floor(angles_normalized * 18)  # angular divisions
    checker_pattern = ((checker_u + checker_v) % 2).astype(float)

    # Apply to image
    image = np.zeros((pixel_height, pixel_width))
    if Checkering == True:
        image[hit_mask] = checker_pattern[hit_mask]
    else:
        image[hit_mask] = 1.0
    

    plt.imshow(image, cmap = 'gray', origin = 'lower')
    
    
    if ShowImage == True:
        plt.show()
    return image

def make_gif(radius, theta_start, phi_start, theta_sweep, phi_sweep, framecount, fps, filename="camera_sweep.gif"):
    
    theta_vals = np.linspace(theta_start, theta_start + theta_sweep, framecount)
    phi_vals = np.linspace(phi_start, phi_sweep, framecount)

    for theta, phi in zip(theta_vals, phi_vals):
        camera_position[:] = sph_to_cart(radius, theta, phi)
        
        rays = make_camera_rays(pixel_width, pixel_height, 
                        fieldofview, focal_length, camera_position)
        image = create_image(rays)

        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray", origin="lower")
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        frames.append(imageio.imread(buf))
        plt.close(fig)
    
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved GIF: {filename}")
        


'''
rays = make_camera_rays(pixel_width=pixel_width, pixel_height=pixel_height, 
                        fieldofview=fieldofview, focal_length=focal_length, 
                        camera_position=camera_position)

create_image(rays)

'''


make_gif(radius=camera_radius,theta_start=camera_theta,phi_start=camera_phi,theta_sweep=theta_rad_sweep,
         phi_sweep=phi_rad_sweep, framecount=framecount, fps=10, filename="camera_sweep.gif")



















#----- GIF creation -----

