

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

#tested and working
def sph_to_cart(r, theta, phi):
    """
    Converts spherical coordinates to Catesian coordinates.
    r: radius
    theta: polar angle (angle from +z axis)
    phi: azimuthal angle (angle in x-y plane from +x axis)
    Returns a numpy array [x, y, z] in Cartesian coordinates.
    """ 

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

#tested and working
def impact_parameter(camera_position, black_hole_position, ray_dir):
    '''
    returns an impact parameter b = || r0 x D ||
    where r0 is the vector from the camera to the black hole, and D is the ray direction.
    Camera position is provided in cartesian coordinates
    Ray direction is a unit vector in the direction of the ray.
    '''
    r0= camera_position - black_hole_position  # vector from BH to camera
    return np.linalg.norm(np.cross(r0, ray_dir))

#not tested, but should work since initial_state_ray uses it
def ortho_basis_for_ray(camera_position, ray_direction):
    """
    Build an orthonormal basis tied to the ray's plane of motion:
      e_r0     : radial unit vector from BH -> camera
      e_theta0 : in-plane, perpendicular to e_r0 (right-hand with n)
      n_hat    : normal to the plane
    """

    e_r0 = camera_position/np.linalg.norm(camera_position)      #radial unit vector
    n = np.cross(camera_position, ray_direction)                #cross of position vector and ray gives vector perpendicular to the plane made by the two
    n_norm = np.linalg.norm(n)

    if n_norm < 1e-12:
        #this is a case for rays being nearly radial, for which we pick an arbitrary perpendicular
        tmp = np.array([1,0,0])
        if abs(np.dot(e_r0,tmp)) > 0.9:
            tmp = np.array([0,1,0])
        
        n = np.cross(e_r0,tmp)
        n_norm = np.linalg.norm(n)

    n_hat = n / n_norm
    e_theta0 = np.cross(n_hat, e_r0)
    return e_r0, e_theta0, n_hat

#No bugs as far as I can tell, but not sure this is the right math.
def initial_state_ray(camera_position, black_hole_position, ray_direction, E=1.0, M=0.05):
    '''
    Generate the starting 4-position, 4-momentum of a ray for integration in Schwarzschild spacetime.
    
    Returns a state array:
    [t, r, theta, phi, p_t, p_r, p_theta, p_phi]
    '''
    t = 0.0
    r = np.linalg.norm(camera_position-black_hole_position)
    
    r0 = r
    theta0 = np.arccos(camera_position[2]/r0)
    phi0 = np.arctan2(camera_position[1], camera_position[0])

    e_r, e_theta, e_phi = ortho_basis_for_ray(camera_position, ray_direction)
    
    # k_r, k_theta, k_phi are unit-vector components: k_r^2 + k_theta^2 + k_phi^2 = 1
    k_r = np.dot(ray_direction, e_r)
    k_theta = np.dot(ray_direction, e_theta)
    k_phi = np.dot(ray_direction, e_phi)

    b = impact_parameter(camera_position,black_hole_position, ray_direction)
    

    # Choose alpha to satisfy the null condition with the chosen p^t:
    alpha = E / np.sqrt(k_r**2 + (1.0 - 2.0*M/r0)*(k_theta**2 + k_phi**2))

    # Contravariant spatial components (drive dr/dλ, dθ/dλ, dφ/dλ)
    pr     = alpha * k_r
    ptheta = alpha * (k_theta / r0)
    pphi   = alpha * (k_phi / (r0 * np.sin(theta0) + 1e-16))  # small eps for pole safety
    
    pt = E / (1 - 2*M/r0)


    return np.array([t,r0,theta0,phi0,pt,pr,ptheta,pphi])

def position_derivatives(state):
    """
    Compute the derivatives of the spacetime coordinates 
    (t, r, theta, phi) along the geodesic.

    Parameters
    ----------
    state : array_like, shape (8,)
        The current state vector:
        [t, r, theta, phi, p_t, p_r, p_theta, p_phi]

    Returns
    -------
    derivs : ndarray, shape (4,)
        The derivatives [dt/dλ, dr/dλ, dθ/dλ, dφ/dλ]
    """
    t, r, th, ph, pt, pr, pth, pph = state
    dt = pt
    dr = pr
    dth = pth / (r*r)
    dph = pph / (r*r*np.sin(th)**2)

    return np.array([dt, dr, dth, dph], dtype=float)

def momentum_derivatives(state, M, eps=1e-12):
    ''' 
    Momentum (contravariant) derivatives for a photon in Schwarzschild spacetime.

    Parameters
    ----------
    state : array_like, shape (8,)
        [t, r, theta, phi, p^t, p^r, p^theta, p^phi]
    M : float
        Black hole mass in geometric units (G=c=1).
    eps : float
        Small regularizer to avoid division by zero near horizon/poles.
    Returns'''

    t, r, theta, phi, pt, pr, pth, pph = state

    # Safeguards near horizon and poles
    r = max(r, 2.0*M + eps) #eps avoids problems at horizon
    f = 1.0 - 2.0*M / r
    sin_th = np.sin(theta) #avoids problems at pole
    cos_th = np.cos(theta)
    sin_th_safe = sin_th if abs(sin_th) > eps else (eps if sin_th >= 0 else -eps)
    cot_th = cos_th / sin_th_safe

    # dp^t/dλ
    dpt = -(2.0*M / (r*r * f)) * pt * pr

    # dp^r/dλ
    dpr = ( - (M * f / (r*r)) * (pt*pt)
            + (M / (r*r * f)) * (pr*pr)
            + (r * f) * (pth*pth)
            + (r * f) * (sin_th*sin_th) * (pph*pph) )

    # dp^θ/dλ
    dpth = -(2.0 / r) * pr * pth + (sin_th * cos_th) * (pph*pph)

    # dp^φ/dλ
    dpph = -(2.0 / r) * pr * pph - 2.0 * cot_th * pth * pph

    return np.array([dpt, dpr, dpth, dpph])

def geodesic_rhs(state, M, eps=1e-12):
    '''
    Full RHS for photon geodesic equations in Schwarzschild spacetime.

    Parameters
    ----------
    state : array_like, shape (8,)
        [t, r, theta, phi, p^t, p^r, p^theta, p^phi]
    M : float
        Black hole mass in geometric units.

    Returns
    -------
    dstate : ndarray, shape (8,)
        Derivatives wrt affine parameter λ:
        [dt/dλ, dr/dλ, dθ/dλ, dφ/dλ, dp^t/dλ, dp^r/dλ, dp^θ/dλ, dp^φ/dλ]
    '''
        
    dt, dr, dth, dph = position_derivatives(state)
    dpt, dpr, dpth, dpph = momentum_derivatives(state, M, eps=eps)

    return np.array([dt, dr, dth, dph, dpt, dpr, dpth, dpph])




def make_camera_rays(pixel_width, pixel_height, fieldofview, focal_length, camera_position,black_hole_position = np.array([0.0, 0.0, 0.0])):

    aspect_ratio = pixel_width / pixel_height
    fov_rad = np.deg2rad(fieldofview)
    plane_height = 2 * focal_length * np.tan(fov_rad / 2)
    plane_width = plane_height * aspect_ratio


    #creating image plane
    xs = np.linspace(-plane_width/2, plane_width/2, pixel_width)
    ys = np.linspace(-plane_height/2, plane_height/2, pixel_height)
    px, py = np.meshgrid(xs, ys)
    pz = np.full_like(px, focal_length)


    #creating rays
    rays = np.stack((px,py,pz),axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays /= norms #normalize rays to represent direction

    #camera look-at rotation
    # Forward vector (from camera to black hole)
    forward = (black_hole_position - camera_position)
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
    
    #Commented out for sanity
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
    
    #normalized radius for post effects
    x = (radii - inner_radius) / (outer_radius - inner_radius)
    x = np.clip(x, 0.0, 1.0)
    falloff = 1.0 - x**3

    shading = checker_pattern if Checkering else np.ones_like(radii)
    image = np.zeros((pixel_height, pixel_width))
    image[hit_mask] = (shading * falloff)[hit_mask]

    
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
        
        fig.set_size_inches(pixel_width / 100, pixel_height / 100)  # assuming 100 dpi
        ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        frames.append(imageio.imread(buf))
        plt.close(fig)
    
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved GIF: {filename}")
