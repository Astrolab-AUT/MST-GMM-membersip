import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import pandas as pd

# Function to compute the Minimum Spanning Tree (MST)
def min_span_tree(df_data, params, num_std=1, manual_th=True, threshold=0.85, y_limit=5, fig=True):
    """
    Computes the Minimum Spanning Tree (MST) for the given dataset.

    Parameters:
        df_data (DataFrame): The input data.
        params (list): List of columns to be used for MST computation.
        num_std (float): Number of standard deviations for threshold calculation.
        manual_th (bool): Whether to use a manual threshold or calculate it automatically.
        threshold (float): The manual threshold value for filtering.
        y_limit (int): Y-axis limit for the edge weight plot.
        fig (bool): Whether to plot the edge weights.

    Returns:
        tuple: Threshold value and updated DataFrame with MST cluster information.
    """
    np.random.seed(101)
    df_data = df_data.reset_index(drop=True)
    data = df_data.copy()
    scaler = StandardScaler()
    data[params] = scaler.fit_transform(data[params])

    nn = NearestNeighbors(n_neighbors=3, metric='minkowski')
    nn.fit(data[params])
    distances, indices = nn.kneighbors(data[params])

    G = nx.Graph()
    n_samples = data.shape[0]
    for i in range(n_samples):
        for j in range(1, len(indices[i])):
            neighbor_index = indices[i, j]
            weight = distances[i, j]
            G.add_edge(i, neighbor_index, weight=weight)

    mst = nx.minimum_spanning_tree(G)
    edge_weights = [X['weight'] for _, _, X in mst.edges(data=True)]

    if not manual_th:
        threshold = np.mean(edge_weights) + num_std * np.std(edge_weights)
        print('threshold:', round(threshold, 2))

    if fig:
        sorted_weights = sorted(edge_weights)
        plt.figure(figsize=(4, 2))
        plt.plot(sorted_weights)
        plt.hlines(threshold, 0, len(df_data), colors='red')
        plt.ylim(0, y_limit)
        plt.xlabel('Edge Index')
        plt.ylabel('Edge Weight')
        plt.title('Sorted Edge Weights')
        plt.show()

    nodes_to_remove = [node for node in mst.nodes() if mst.degree(node, weight='weight') > threshold]
    mst.remove_nodes_from(nodes_to_remove)

    df_data['MST_cluster'] = 1
    df_data.loc[nodes_to_remove, 'MST_cluster'] = 0

    return round(threshold, 2), df_data

# Function to preprocess the cluster data
def preprocess_cluster(data, g_mean_th=19):
    """
    Filters and preprocesses the cluster data based on parallax and photometric magnitude.

    Parameters:
        data (DataFrame): Input DataFrame.
        g_mean_th (float): Threshold for photometric G mean magnitude.

    Returns:
        DataFrame: Preprocessed data.
    """
    data = data[data['parallax'] > 0]
    data = data[data['phot_g_mean_mag'] < g_mean_th]
    data['Gmg'] = data['phot_g_mean_mag'] + (5 * np.log10(data['parallax']) - 10)
    data['L'] = 10 ** (0.4 * (4.83 - data['Gmg']))
    data = data[(abs(data['pmra']) < 10) & (abs(data['pmdec']) < 10)]
    print(len(data))
    return data

# Function to plot CMD using scatterplot
def cmd_plot(data, x_axis, y_axis, alpha=0.8, s=5):
    """
    Plots a Color-Magnitude Diagram (CMD).

    Parameters:
        data (DataFrame): Input data.
        x_axis (str): Column name for x-axis.
        y_axis (str): Column name for y-axis.
        alpha (float): Transparency of points.
        s (int): Size of points.
    """
    plt.figure(figsize=(6, 4), dpi=100)
    sns.scatterplot(data=data, y=y_axis, x=x_axis, alpha=alpha, s=s, color='black', edgecolors='none', linewidth=0)
    plt.gca().invert_yaxis()
    plt.show()

# Function to create a joint plot
def joint_plot(data):
    """
    Creates a joint KDE plot for proper motion data.

    Parameters:
        data (DataFrame): Input data with 'pmra' and 'pmdec' columns.
    """
    plt.figure(dpi=90)
    sns.jointplot(data=data, x="pmra", y="pmdec", kind="kde")
    plt.show()

# Function to fit a Gaussian curve to histogram data
def fit_curve(data, column, bins=100):
    """
    Fits a Gaussian curve to the histogram of the specified column.

    Parameters:
        data (DataFrame): Input data.
        column (str): Column name for histogram.
        bins (int): Number of bins.

    Returns:
        tuple: Optimal parameters for the Gaussian curve.
    """
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    counts, bins, _ = plt.hist(data[column], bins=bins)
    x_data = bins[:-1]
    y_data = counts

    popt, _ = curve_fit(gaussian, x_data, y_data, maxfev=5000)

    plt.figure(figsize=(8, 2), dpi=80)
    sns.scatterplot(x=x_data, y=y_data, label=column)
    plt.plot(x_data, gaussian(x_data, *popt), color='red', label='Fit')
    plt.legend()
    plt.show()

    return popt

# Function to filter data using Gaussian bounds
def guassian_filter(data, column, mu, std):
    """
    Filters data based on Gaussian bounds.

    Parameters:
        data (DataFrame): Input data.
        column (str): Column name to filter.
        mu (float): Mean value for the Gaussian.
        std (float): Standard deviation for the Gaussian.

    Returns:
        DataFrame: Filtered data.
    """
    up = round(mu + 3 * std, 2)
    low = round(mu - 3 * std, 2)

    if up > low:
        print(f'{column} upper bound:', up)
        print(f'{column} lower bound:', low)
        df = data[(data[column] < up) & (data[column] > low)]
    else:
        print(f'{column} upper bound:', low)
        print(f'{column} lower bound:', up)
        df = data[(data[column] < low) & (data[column] > up)]

    print('cluster length:', len(df))
    return df

# Function to compute luminosity density
def luminosity_density(cluster_3d, clusterdf):
    """
    Computes the luminosity density for the given cluster data.

    Parameters:
        cluster_3d (DataFrame): 3D positions of the cluster members.
        clusterdf (DataFrame): DataFrame containing luminosity information.

    Returns:
        ndarray: Luminosity density values.
    """
    nbrs = NearestNeighbors(n_neighbors=6, metric='minkowski').fit(cluster_3d)
    distances, indices = nbrs.kneighbors(cluster_3d)
    max_distances = np.amax(distances, axis=1)
    spheres = (4 / 3) * np.pi * (max_distances ** 3)

    lum_sum = [np.sum(clusterdf.iloc[indices[i]]['L']) for i in range(len(clusterdf))]
    lum_dens = np.array(lum_sum) / spheres

    return lum_dens

# Function to plot luminosity density profile
def lum_plot(data):
    """
    Plots the luminosity density profile.

    Parameters:
        data (ndarray): Luminosity density values.
    """
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(range(len(data)), np.sort(data))
    plt.ylabel('ΔL/ΔV')
    plt.title('Luminosity Density Profile')
    plt.show()

# Function to plot CMD with predefined markers
def cmd_plotly(data, x_axis, y_axis, huex='cluster', ax=None, alpha=0.8, s=7, theme=None, markers=['o', 'x']):
    """
    Plots a Color-Magnitude Diagram (CMD) with predefined markers for clusters.

    Parameters:
        data (DataFrame): Input data.
        x_axis (str): Column name for x-axis.
        y_axis (str): Column name for y-axis.
        huex (str): Column for hue (e.g., cluster).
        ax (Axes): Matplotlib axes object (optional).
        alpha (float): Transparency of points.
        s (int): Size of points.
        theme (list): Colors for clusters.
        markers (list): Markers for clusters.
    """
    with plt.style.context(['ieee']):
        if ax is None:
            fig = plt.figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)

        cluster_values = [0, 1]
        for cluster_value, marker in zip(cluster_values, markers):
            cluster_data = data[data[huex] == cluster_value]
            sns.scatterplot(
                data=cluster_data,
                x=x_axis,
                y=y_axis,
                alpha=alpha,
                s=s,
                marker=marker,
                ax=ax,
                color=theme[cluster_value],
                edgecolor='none' if marker == 'x' else theme[cluster_value],
                facecolor='none' if marker == 'o' else theme[cluster_value],
                label=f'Cluster {cluster_value}'
            )

        ax.invert_yaxis()
        ax.legend()

def king_profile_function(r, f_b, f_0, R_C):
    """
    King profile function:
    f(r) = f_b + f_0 / (1 + (r / R_C)^2)
    """
    return f_b + f_0 / (1 + (r / R_C)**2)

def fit_king_profile(data, path, radius_num=50, cluster_name="Cluster", plotting=True):
    """
    Fit the King profile to star cluster data.
    Parameters:
        - data: pandas DataFrame with 'ra' and 'dec' columns for star positions.
        - radius_num: Number of radial bins.
        - cluster_name: Name of the cluster for title and file saving.
        - plotting: Whether to plot the fit.
    Returns:
        - coefs: Best-fit parameters [f_b, f_0, R_C].
        - R_tidal: Calculated tidal radius.
        - cov: Covariance matrix of the fit.
    """
    # Determine the center of the cluster
    center = (np.mean(data['ra']), np.mean(data['dec']))
    max_r = round(max(np.linalg.norm(center - data[['ra', 'dec']], axis=1)), 2) + 0.01

    # Define radial bins
    radii = np.linspace(0, max_r, radius_num)
    x = np.linspace(0, max_r, num=100)

    # Calculate star densities
    star_densities = []
    for i in range(radius_num - 1):
        r_inner = radii[i]
        r_outer = radii[i + 1]
        distances = np.sqrt((data['ra'] - center[0])**2 + (data['dec'] - center[1])**2)
        stars_within_circle = data[(distances >= r_inner) & (distances < r_outer)]
        star_count = len(stars_within_circle)
        circle_area = np.pi * (r_outer**2 - r_inner**2)
        star_density = star_count / circle_area if circle_area > 0 else 0
        star_densities.append(star_density)

    star_densities = np.array(star_densities)

    # Fit King profile
    try:
        coefs, cov = curve_fit(king_profile_function, radii[:-1], star_densities, maxfev=5000, bounds=[0, np.inf])
    except RuntimeError:
        print("Curve fitting failed. Returning default values.")
        return None, None, None

    # Calculate tidal radius directly
    sigma_b = np.sqrt(np.diag(cov))[0]  # Uncertainty in f_b
    term = coefs[1] / (3 * sigma_b) - 1  # f_0 / (3 * sigma_b) - 1
    if term > 0:
        R_tidal = coefs[2] * np.sqrt(term)
    else:
        print("Warning: Invalid values for tidal radius calculation.")
        R_tidal = None

    # Plotting the results
    if plotting:
        with plt.style.context(['science','ieee', 'no-latex']):
            plt.figure(figsize=(5, 5), dpi=300)
            plt.bar(radii[:-1], star_densities, width=radii[1]-radii[0], color='black', alpha=0.9, label='Data')
            plt.plot(x, king_profile_function(x, coefs[0], coefs[1], coefs[2]), '--', color='red', lw=1.7, label='King Profile')
            plt.legend()
            plt.xlabel(r'$r$ (arcmin)')
            plt.ylabel(r'$\rho$ (stars arcmin$^{-2}$)')
            plt.title(f"{cluster_name}")
            plt.savefig(path+f"/{cluster_name.replace(' ', '_')}_kp.pdf")
            plt.savefig(path+f"/{cluster_name.replace(' ', '_')}_kp.jpg")
            plt.show()

    return coefs, R_tidal, cov