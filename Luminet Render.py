

#to-do:
#update to use full GR DEs on rays within a certain radius of the BH
#add coloration to accretion disk

#when implementing full null geodesics, have precalculator and adaptive step size to speed up ray tracing
#maybe shortcut the rendering process of GIFs by flipping the images instead of recalculating the rays

from Functions import sph_to_cart, ortho_basis_for_ray, impact_parameter, initial_state_ray, make_camera_rays, create_image, make_gif
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

#these boolean toggles may get removed in the future
Testmode = False    #test mode toggles extra print statements for debugging/sanity checks, ray visualization 
Checkering = True   #creates a checkerboard pattern on the disk to better visualize distortion

#if one is True, other must be False
ShowImage = False   #if True, will show the final image after ray tracing
Gifmaker = False    #if True, will create a gif of the camera orienting from above to below the black hole

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



#camera position in Cartesian coordinates
camera_cart = sph_to_cart(camera_radius, camera_theta,camera_phi)
x_cam = bh_pos[0] + camera_cart[0]
y_cam = bh_pos[1] + camera_cart[1]
z_cam = bh_pos[2] + camera_cart[2]
camera_position = np.array([x_cam, y_cam, z_cam]) #relative to black hole position



if Gifmaker == True:
    # Create the GIF with camera sweep
    print("Creating GIF...")
    make_gif(radius=camera_radius,theta_start=camera_theta,phi_start=camera_phi,theta_sweep=theta_rad_sweep,
            phi_sweep=phi_rad_sweep, framecount=framecount, fps=10, filename="camera_sweep.gif")



camera_position = np.array([10.0, 10.0, 10.0]) #test position
black_hole_position = np.array([0.0, 0.0, 0.0]) #test position

print(initial_state_ray(camera_position, black_hole_position, ray_direction=-camera_position/np.linalg.norm(camera_position)))
















