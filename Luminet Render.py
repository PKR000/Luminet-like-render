

#to-do:
#update to use full GR DEs on rays within a certain radius of the BH
#add coloration to accretion disk

#when implementing full null geodesics, have precalculator and adaptive step size to speed up ray tracing
#maybe shortcut the rendering process of GIFs by flipping the images instead of recalculating the rays

from Functions import sph_to_cart, initial_state_ray, make_gif, geodesic_rhs, integrate_single_ray_geodesic
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io
from scipy.integrate import solve_ivp

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



camera_position = np.array([10.0, 10.0, 0.0]) #test position
black_hole_position = np.array([0.0, 0.0, 0.0]) #test position
ray_direction = np.array([1.0, 0.0, 0.0])

# Initial state: [t, r, θ, φ, pt, pr, pθ, pφ]
state0 = initial_state_ray(camera_position, black_hole_position, ray_direction, M=1.0, E=1.0)
testcase = [0.0, 10.0, np.pi/2, 0.0, 1.0, 0.0, 0.0, 0.01] #test case for initial state
print(state0)

trajectory = integrate_single_ray_geodesic(testcase, h=0.01, nsteps=5000)

r_vals = trajectory[:,1]
phi_vals = trajectory[:,3]
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x_vals, y_vals, 'b-', lw=1.2, label="Photon path")
ax.add_patch(plt.Circle((0,0), 2.0, color='k'))  # event horizon at r=2M
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Photon trajectory around Schwarzschild BH")
ax.set_aspect("equal")
ax.legend()
plt.show()