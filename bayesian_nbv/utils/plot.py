import numpy as np
# import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_1d_data(x_query, x_data, y_data, f_gt, s=None, alpha=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_data, y_data, color='navy', label='Data Points', s=s, alpha=alpha)
    ax.plot(x_query, f_gt, color='gray', label='Ground Truth')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.legend()
    plt.grid()
    plt.show()


def plot_1d(x_query, x_data, y_data, f, f_gt, std, s=None, alpha=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_data, y_data, color='navy', label='Data Points', s=s, alpha=alpha)
    ax.plot(x_query, f, color='blue', label='GP Mean')
    if f_gt is not None:
        ax.plot(x_query, f_gt, color='gray', label='Ground Truth')
    l = f - 1.96 * std
    u = f + 1.96 * std
    ax.fill_between(x_query.flatten(), l.flatten(), u.flatten(), color='lightblue', alpha=0.5, label='95% Confidence Interval')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('1D Gaussian Process Regression')
    ax.legend()
    plt.grid()
    plt.show()

def plot_1d_class(x_query, x_data, y_data, y_predict, inducing_points, title='1D SVGP Classification', s=None, alpha=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_data, y_data, color='navy', label='Data Points', s=s, alpha=alpha)
    ax.plot(x_query, y_predict, color='blue', label='P(y = 1 | X)')
    y_std = jnp.sqrt(y_predict * (1 - y_predict)) # Standard deviation for binary classification
    l = y_predict - y_std
    u = y_predict + y_std
    ax.fill_between(x_query.flatten(), l.flatten(), u.flatten(), color='lightblue', alpha=0.5, label='±Std Confidence Interval')

    for xi in inducing_points:
        ax.axvline(x=xi, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title(title)
    # ax.set_ylim(-0.5, 1.5)
    ax.legend()
    plt.grid()
    plt.show()


def plot_1d_svgp(x_query, x_data, y_data, f, f_gt, std, inducing_points, inducing_mean, title='1D SVGP Regression', s=None, alpha=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_data, y_data, color='navy', label='Data Points', s=s, alpha=alpha)
    ax.plot(x_query, f, color='blue', label='GP Mean')
    if f_gt is not None:
        ax.plot(x_query, f_gt, color='gray', label='Ground Truth')
    if std is not None:
        l = f - 1.96 * std
        u = f + 1.96 * std
        ax.fill_between(x_query.flatten(), l.flatten(), u.flatten(), color='lightblue', alpha=0.5, label='95% Confidence Interval')
    if inducing_mean is not None:
        ax.scatter(inducing_points, inducing_mean, marker='x', color='red', label='Inducing Points')
    else:
        for xi in inducing_points:
            ax.axvline(x=xi, color='red', linestyle='--', alpha=0.5, label='Inducing Points')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title(title)
    # ax.set_ylim(-0.5, 1.5)
    ax.legend()
    plt.grid()
    plt.show()

def plot_2d_updated_data(x1, y1, x2, y2, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x1, y1, color='red', alpha=0.5, s=10, marker='o', label='Original Points')
    ax.scatter(x2, y2, color='blue', alpha=0.8, s=20, marker='x', label='Updated Points')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if title is not None:
        ax.set_title(title)
    ax.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()

def plot_2d_data(x, y, u, v, title=None, xlim=None, ylim=None, last_new=False):
    fig, ax = plt.subplots(figsize=(8, 8))

    if last_new:
        ax.scatter(x[:-1], y[:-1], color='red', marker='.', alpha=0.5)
        ax.quiver(x[:-1], y[:-1], u[:-1], v[:-1], color='red', width=0.003, scale=25, alpha=0.5)
        ax.scatter(x[-1:], y[-1:], color='blue', marker='x', alpha=0.5)
        ax.quiver(x[-1:], y[-1:], u[-1:], v[-1:], color='blue', width=0.003, scale=25, alpha=1.0)
    else:
        ax.scatter(x, y, color='red', alpha=0.5)
        ax.quiver(x, y, u, v, color='red', width=0.003, scale=25, alpha=0.5)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if title is not None:
        ax.set_title(title)
    plt.axis("equal")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    plt.grid()
    plt.show()

def plot_2d_vec(x, y, u, v, variance=None, x_points=None, y_points=None, u_points=None, v_points=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    # M = np.sqrt(u**2 + v**2)
    if variance is not None:
        q = ax.quiver(x, y, u, v, variance, cmap='plasma', width=0.003, scale=25, zorder=5)
        plt.colorbar(q, ax=ax, label='Variance')
    else:
        ax.quiver(x, y, u, v, color='blue', width=0.003, scale=25)
    if x_points is not None and y_points is not None and u_points is not None and v_points is not None:
        ax.scatter(x_points, y_points, color='red', label='Point cloud', alpha=0.5)
        ax.quiver(x_points, y_points, u_points, v_points, color='red', width=0.003, scale=25, alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if title is not None:
        ax.set_title(title)
    plt.axis("equal")
    plt.grid()
    plt.show()

def plot_2d_variance(x, y, var, x_points=None, y_points=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.pcolormesh(x, y, var, shading='Gouraud', cmap='plasma')
    if x_points is not None and y_points is not None:
        ax.scatter(x_points, y_points, marker='x', color='green', label='Point cloud', alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if title is not None:
        ax.set_title(title)
    plt.axis("equal")
    plt.colorbar(c, ax=ax)
    plt.grid()
    plt.show()

def plot_2d_vec_svgp(x, y, u, v, variance, inducing_xy, inducing_uv, x_points=None, y_points=None, u_points=None, v_points=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    # M = np.sqrt(u**2 + v**2)
    # if variance is not None:
    # print(x.shape, y.shape, u.shape, v.shape, variance.shape)
    q = ax.quiver(x, y, u, v, variance, cmap='plasma', width=0.003, scale=25, zorder=4)
    plt.colorbar(q, ax=ax, label='Variance')
    # else:
    #     ax.quiver(x, y, u, v, color='blue', width=0.003, scale=25)
    ax.scatter(inducing_xy[:, 0], inducing_xy[:, 1], marker='x', color='green', label='Inducing Points', zorder=5)
    ax.quiver(inducing_xy[:, 0], inducing_xy[:, 1], inducing_uv[:, 0], inducing_uv[:, 1], color='green', width=0.003, scale=25, zorder=5)
    if x_points is not None and y_points is not None and u_points is not None and v_points is not None:
        ax.scatter(x_points, y_points, color='red', label='Point cloud', alpha=0.5)
        ax.quiver(x_points, y_points, u_points, v_points, color='red', width=0.003, scale=25, alpha=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if title is not None:
        ax.set_title(title)
    plt.axis("equal")
    plt.grid()
    plt.show()

def plot_mean(name, x1, x2, f, title="", xlim=None, ylim=None, save=True):
    plt.figure(figsize=(8, 8))
    max_abs_val = np.abs(f).max()
    
    # Create the pcolormesh plot
    im = plt.pcolormesh(x1, x2, f.reshape(x1.shape), shading="Gouraud", cmap="RdBu", vmin=-max_abs_val, vmax=max_abs_val)
    
    # Add colorbar
    plt.colorbar(im, shrink=0.8, aspect=20)
    
    plt.contour(x1, x2, f.reshape(x1.shape), levels=5, colors="k")
    plt.contour(x1, x2, f.reshape(x1.shape), np.array([0.0]), colors="limegreen")
    plt.axis("equal")
    # plt.axis("off")
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.grid()
    if title:
        plt.title(title)
    
    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=400)
    else:
        plt.show()

def plot_mean_2d_svgp(name, x1, x2, f, inducing_xy, inducing_uv, x_points=None, y_points=None, title="", save=True):
    plt.figure(figsize=(8, 8))
    max_abs_val = np.abs(f).max()
    
    # Create the pcolormesh plot
    im = plt.pcolormesh(x1, x2, f.reshape(x1.shape), shading="Gouraud", cmap="RdBu", vmin=-max_abs_val, vmax=max_abs_val)
    
    # Add colorbar
    plt.colorbar(im, shrink=0.8, aspect=20)
    
    plt.contour(x1, x2, f.reshape(x1.shape), levels=5, colors="k")
    plt.contour(x1, x2, f.reshape(x1.shape), np.array([0.0]), colors="limegreen")

    plt.scatter(inducing_xy[:, 0], inducing_xy[:, 1], marker='x', color='green', label='Inducing Points', zorder=5)
    plt.quiver(inducing_xy[:, 0], inducing_xy[:, 1], inducing_uv[:, 0], inducing_uv[:, 1], color='green', width=0.003, scale=25, zorder=5)

    if x_points is not None and y_points is not None:
        plt.scatter(x_points, y_points, color='red', label='Point cloud', alpha=0.5)

    plt.axis("equal")
    # plt.axis("off")
    plt.grid()
    if title:
        plt.title(title)
    
    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=400)
    else:
        plt.show()

def plot_variance(name, x1, x2, f, save=True):
    plt.figure(figsize=(4, 4))
    plt.pcolormesh(x1, x2, f.reshape(x1.shape), shading="Gouraud", cmap="plasma")
    plt.axis("equal")
    plt.axis("off")
    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=400)
    else:
        plt.show()