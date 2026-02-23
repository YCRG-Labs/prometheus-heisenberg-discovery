"""
Visualization Module

This module implements comprehensive visualization functions for the J1-J2 Heisenberg
Prometheus framework, including phase diagrams, latent space trajectories, correlation
heatmaps, critical point detection plots, scaling collapse plots, and training curves.

All plots are generated with publication-quality settings and include uncertainty bands
where appropriate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set publication-quality plot defaults
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class Visualizer:
    """
    Comprehensive visualization for J1-J2 Heisenberg Prometheus analysis.
    
    Generates publication-quality plots for:
    - Phase diagrams showing order parameters vs frustration ratio
    - Latent space trajectories with color coding
    - Correlation heatmaps between latent dimensions and observables
    - Critical point detection showing all methods
    - Finite-size scaling collapse with optimized exponents
    - Q-VAE training curves
    
    All plots include uncertainty bands from bootstrap analysis where available.
    """
    
    def __init__(self, config: Any):
        """
        Initialize Visualizer.
        
        Args:
            config: Configuration object with output_dir parameter
        """
        self.config = config
        self.output_dir = Path(getattr(config, 'output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different plot types
        self.phase_diagram_dir = self.output_dir / 'phase_diagrams'
        self.latent_dir = self.output_dir / 'latent_space'
        self.correlation_dir = self.output_dir / 'correlations'
        self.critical_point_dir = self.output_dir / 'critical_points'
        self.scaling_dir = self.output_dir / 'scaling'
        self.training_dir = self.output_dir / 'training'
        
        for directory in [self.phase_diagram_dir, self.latent_dir, 
                         self.correlation_dir, self.critical_point_dir,
                         self.scaling_dir, self.training_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Visualizer with output directory: {self.output_dir}")
    
    def plot_phase_diagram(self,
                          observables: pd.DataFrame,
                          observable_names: Optional[List[str]] = None,
                          save_name: str = 'phase_diagram.png') -> None:
        """
        Plot order parameters vs j2_j1 for all lattice sizes.
        
        Creates a multi-panel figure showing how different observables vary
        with the frustration ratio j2_j1. Each panel shows data for all
        lattice sizes with different colors/markers.
        
        Args:
            observables: DataFrame with columns [j2_j1, L, observable_name, value]
                        or wide format with observable columns
            observable_names: List of observable names to plot (default: all)
            save_name: Filename for saved plot
        """
        logger.info("Generating phase diagram plot...")
        
        # Convert to wide format if needed
        if 'observable_name' in observables.columns and 'value' in observables.columns:
            obs_wide = observables.pivot_table(
                index=['j2_j1', 'L'],
                columns='observable_name',
                values='value'
            ).reset_index()
        else:
            obs_wide = observables.copy()
        
        # Determine which observables to plot
        if observable_names is None:
            observable_names = [col for col in obs_wide.columns 
                              if col not in ['j2_j1', 'L']]
        
        # Filter to requested observables
        observable_names = [name for name in observable_names if name in obs_wide.columns]
        
        if not observable_names:
            logger.warning("No valid observables found for phase diagram")
            return
        
        # Create figure with subplots
        n_obs = len(observable_names)
        n_cols = min(3, n_obs)
        n_rows = (n_obs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_obs == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Get unique lattice sizes
        lattice_sizes = sorted(obs_wide['L'].unique())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(lattice_sizes)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        # Plot each observable
        for idx, obs_name in enumerate(observable_names):
            ax = axes[idx]
            
            for i, L in enumerate(lattice_sizes):
                # Filter data for this lattice size
                data_L = obs_wide[obs_wide['L'] == L].sort_values('j2_j1')
                
                if len(data_L) == 0:
                    continue
                
                j2_j1 = data_L['j2_j1'].values
                obs_values = data_L[obs_name].values
                
                # Plot with error bars if available
                marker = markers[i % len(markers)]
                ax.plot(j2_j1, obs_values, 
                       marker=marker, markersize=6, 
                       color=colors[i], 
                       label=f'L={L}',
                       linewidth=1.5, alpha=0.8)
            
            # Formatting
            ax.set_xlabel('$J_2/J_1$')
            ax.set_ylabel(obs_name.replace('_', ' ').title())
            ax.set_title(obs_name.replace('_', ' ').title())
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Add vertical lines for known phase boundaries (approximate)
            ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Hide unused subplots
        for idx in range(n_obs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Phase Diagram: Order Parameters vs Frustration Ratio', 
                    fontsize=18, y=1.00)
        plt.tight_layout()
        
        # Save figure
        save_path = self.phase_diagram_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved phase diagram to {save_path}")
        
        plt.close()

    def plot_latent_trajectories(self,
                                latent_reps: Dict[Tuple[float, int], np.ndarray],
                                projection_method: str = 'pca',
                                color_by: str = 'j2_j1',
                                save_name: str = 'latent_trajectories.png') -> None:
        """
        Plot latent space trajectories colored by j2_j1 or L.
        
        Projects high-dimensional latent space to 2D/3D for visualization.
        Trajectories show how latent representations evolve with parameters.
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            projection_method: Method for dimensionality reduction ('pca', 'tsne', 'umap')
            color_by: Variable to use for coloring ('j2_j1' or 'L')
            save_name: Filename for saved plot
        """
        logger.info(f"Generating latent trajectory plot with {projection_method}...")
        
        if not latent_reps:
            logger.warning("No latent representations provided")
            return
        
        # Prepare data
        keys = list(latent_reps.keys())
        j2_j1_values = np.array([k[0] for k in keys])
        L_values = np.array([k[1] for k in keys])
        latent_vectors = np.array([latent_reps[k] for k in keys])
        
        # Perform dimensionality reduction
        if projection_method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            latent_2d = reducer.fit_transform(latent_vectors)
            explained_var = reducer.explained_variance_ratio_
            logger.info(f"PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
        elif projection_method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(keys)//2))
            latent_2d = reducer.fit_transform(latent_vectors)
        elif projection_method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                latent_2d = reducer.fit_transform(latent_vectors)
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                latent_2d = reducer.fit_transform(latent_vectors)
        else:
            raise ValueError(f"Unknown projection method: {projection_method}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine coloring
        if color_by == 'j2_j1':
            color_values = j2_j1_values
            cmap = 'viridis'
            cbar_label = '$J_2/J_1$'
        elif color_by == 'L':
            color_values = L_values
            cmap = 'plasma'
            cbar_label = 'Lattice Size L'
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")
        
        # Plot trajectories for each lattice size
        unique_L = sorted(np.unique(L_values))
        
        for L in unique_L:
            mask = L_values == L
            x = latent_2d[mask, 0]
            y = latent_2d[mask, 1]
            c = color_values[mask]
            
            # Sort by j2_j1 for proper trajectory
            sort_idx = np.argsort(j2_j1_values[mask])
            x = x[sort_idx]
            y = y[sort_idx]
            c = c[sort_idx]
            
            # Plot trajectory line
            ax.plot(x, y, '-', alpha=0.3, linewidth=1, color='gray')
            
            # Plot points with color
            scatter = ax.scatter(x, y, c=c, cmap=cmap, s=100, 
                               alpha=0.8, edgecolors='black', linewidth=0.5,
                               label=f'L={L}' if color_by == 'L' else None)
        
        # Add colorbar
        if color_by == 'j2_j1':
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(cbar_label, fontsize=14)
        
        # Formatting
        ax.set_xlabel(f'{projection_method.upper()} Component 1', fontsize=14)
        ax.set_ylabel(f'{projection_method.upper()} Component 2', fontsize=14)
        ax.set_title(f'Latent Space Trajectories (colored by {cbar_label})', fontsize=16)
        
        if color_by == 'L':
            ax.legend(loc='best', framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        
        # Save figure
        save_path = self.latent_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved latent trajectory plot to {save_path}")
        
        plt.close()
    
    def plot_correlation_heatmap(self,
                                correlations: pd.DataFrame,
                                save_name: str = 'correlation_heatmap.png',
                                vmin: float = -1.0,
                                vmax: float = 1.0,
                                annotate: bool = True) -> None:
        """
        Plot heatmap of correlations between latent dimensions and observables.
        
        Creates a color-coded matrix showing Pearson correlation coefficients
        between each latent dimension and each physical observable.
        
        Args:
            correlations: DataFrame with shape (n_latent_dims, n_observables)
            save_name: Filename for saved plot
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            annotate: Whether to annotate cells with correlation values
        """
        logger.info("Generating correlation heatmap...")
        
        if correlations.empty:
            logger.warning("Empty correlation matrix provided")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(correlations.columns)*0.8), 
                                       max(8, len(correlations.index)*0.5)))
        
        # Create heatmap
        sns.heatmap(correlations, 
                   annot=annotate and len(correlations) * len(correlations.columns) <= 100,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   vmin=vmin,
                   vmax=vmax,
                   cbar_kws={'label': 'Pearson Correlation Coefficient'},
                   linewidths=0.5,
                   linecolor='gray',
                   ax=ax)
        
        # Formatting
        ax.set_xlabel('Physical Observables', fontsize=14)
        ax.set_ylabel('Latent Dimensions', fontsize=14)
        ax.set_title('Latent-Observable Correlation Matrix', fontsize=16)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.correlation_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
        
        plt.close()

    def plot_critical_point_detection(self,
                                     detection_results: Dict[str, Any],
                                     latent_variance_data: Optional[Dict[float, float]] = None,
                                     reconstruction_error_data: Optional[Dict[Tuple[float, int], float]] = None,
                                     fidelity_susceptibility_data: Optional[Dict[Tuple[float, int], float]] = None,
                                     save_name: str = 'critical_point_detection.png') -> None:
        """
        Plot all critical point detection methods on a single figure.
        
        Shows the detection curves for latent variance, reconstruction error,
        and fidelity susceptibility methods, with detected critical points marked.
        
        Args:
            detection_results: Dictionary with keys:
                - 'latent_variance': (j2_j1_c, uncertainty)
                - 'reconstruction_error': (j2_j1_c, uncertainty)
                - 'fidelity_susceptibility': (j2_j1_c, uncertainty)
                - 'ensemble': (j2_j1_c, uncertainty) [optional]
            latent_variance_data: Dict mapping j2_j1 -> variance
            reconstruction_error_data: Dict mapping (j2_j1, L) -> error
            fidelity_susceptibility_data: Dict mapping (j2_j1, L) -> susceptibility
            save_name: Filename for saved plot
        """
        logger.info("Generating critical point detection plot...")
        
        # Create figure with subplots
        n_methods = sum([
            latent_variance_data is not None,
            reconstruction_error_data is not None,
            fidelity_susceptibility_data is not None
        ])
        
        if n_methods == 0:
            logger.warning("No detection data provided")
            return
        
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))
        if n_methods == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # Plot latent variance method
        if latent_variance_data is not None:
            ax = axes[ax_idx]
            ax_idx += 1
            
            j2_j1_vals = np.array(sorted(latent_variance_data.keys()))
            variance_vals = np.array([latent_variance_data[j] for j in j2_j1_vals])
            
            ax.plot(j2_j1_vals, variance_vals, 'o-', color='blue', 
                   linewidth=2, markersize=6, label='Latent Variance')
            
            # Mark detected critical point
            if 'latent_variance' in detection_results:
                j2_j1_c, uncertainty = detection_results['latent_variance']
                ax.axvline(j2_j1_c, color='red', linestyle='--', linewidth=2,
                          label=f'$j_{{2}}/j_{{1}}^{{c}}$ = {j2_j1_c:.3f} ± {uncertainty:.3f}')
                ax.axvspan(j2_j1_c - uncertainty, j2_j1_c + uncertainty,
                          alpha=0.2, color='red')
            
            ax.set_xlabel('$J_2/J_1$', fontsize=14)
            ax.set_ylabel('Latent Variance', fontsize=14)
            ax.set_title('Latent Variance Method', fontsize=16)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        # Plot reconstruction error method
        if reconstruction_error_data is not None:
            ax = axes[ax_idx]
            ax_idx += 1
            
            # Average over lattice sizes
            error_by_j2j1 = {}
            for (j2_j1, L), error in reconstruction_error_data.items():
                if j2_j1 not in error_by_j2j1:
                    error_by_j2j1[j2_j1] = []
                error_by_j2j1[j2_j1].append(error)
            
            j2_j1_vals = np.array(sorted(error_by_j2j1.keys()))
            error_vals = np.array([np.mean(error_by_j2j1[j]) for j in j2_j1_vals])
            error_std = np.array([np.std(error_by_j2j1[j]) for j in j2_j1_vals])
            
            ax.plot(j2_j1_vals, error_vals, 'o-', color='green',
                   linewidth=2, markersize=6, label='Reconstruction Error')
            ax.fill_between(j2_j1_vals, error_vals - error_std, error_vals + error_std,
                           alpha=0.3, color='green')
            
            # Mark detected critical point
            if 'reconstruction_error' in detection_results:
                j2_j1_c, uncertainty = detection_results['reconstruction_error']
                ax.axvline(j2_j1_c, color='red', linestyle='--', linewidth=2,
                          label=f'$j_{{2}}/j_{{1}}^{{c}}$ = {j2_j1_c:.3f} ± {uncertainty:.3f}')
                ax.axvspan(j2_j1_c - uncertainty, j2_j1_c + uncertainty,
                          alpha=0.2, color='red')
            
            ax.set_xlabel('$J_2/J_1$', fontsize=14)
            ax.set_ylabel('Reconstruction Error', fontsize=14)
            ax.set_title('Reconstruction Error Method', fontsize=16)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        # Plot fidelity susceptibility method
        if fidelity_susceptibility_data is not None:
            ax = axes[ax_idx]
            ax_idx += 1
            
            # Average over lattice sizes
            susc_by_j2j1 = {}
            for (j2_j1, L), susc in fidelity_susceptibility_data.items():
                if j2_j1 not in susc_by_j2j1:
                    susc_by_j2j1[j2_j1] = []
                susc_by_j2j1[j2_j1].append(susc)
            
            j2_j1_vals = np.array(sorted(susc_by_j2j1.keys()))
            susc_vals = np.array([np.mean(susc_by_j2j1[j]) for j in j2_j1_vals])
            susc_std = np.array([np.std(susc_by_j2j1[j]) for j in j2_j1_vals])
            
            ax.plot(j2_j1_vals, susc_vals, 'o-', color='purple',
                   linewidth=2, markersize=6, label='Fidelity Susceptibility')
            ax.fill_between(j2_j1_vals, susc_vals - susc_std, susc_vals + susc_std,
                           alpha=0.3, color='purple')
            
            # Mark detected critical point
            if 'fidelity_susceptibility' in detection_results:
                j2_j1_c, uncertainty = detection_results['fidelity_susceptibility']
                ax.axvline(j2_j1_c, color='red', linestyle='--', linewidth=2,
                          label=f'$j_{{2}}/j_{{1}}^{{c}}$ = {j2_j1_c:.3f} ± {uncertainty:.3f}')
                ax.axvspan(j2_j1_c - uncertainty, j2_j1_c + uncertainty,
                          alpha=0.2, color='red')
            
            ax.set_xlabel('$J_2/J_1$', fontsize=14)
            ax.set_ylabel('Fidelity Susceptibility', fontsize=14)
            ax.set_title('Fidelity Susceptibility Method', fontsize=16)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Critical Point Detection Methods', fontsize=18, y=1.00)
        plt.tight_layout()
        
        # Save figure
        save_path = self.critical_point_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved critical point detection plot to {save_path}")
        
        plt.close()

    def plot_scaling_collapse(self,
                             scaling_results: Dict[str, Any],
                             j2_j1: np.ndarray,
                             L: np.ndarray,
                             observable: np.ndarray,
                             observable_name: str = 'Observable',
                             save_name: str = 'scaling_collapse.png') -> None:
        """
        Plot finite-size scaling collapse with optimized exponents.
        
        Shows both raw data and scaled data demonstrating the quality of
        the scaling collapse. Includes uncertainty bands from bootstrap.
        
        Args:
            scaling_results: Dictionary with keys:
                - 'j2_j1_c': Critical point
                - 'nu': Correlation length exponent
                - 'x_O': Scaling dimension
                - 'chi_squared': Collapse quality metric
                - Optional bootstrap results with uncertainties
            j2_j1: Array of frustration ratio values
            L: Array of lattice sizes (same shape as j2_j1)
            observable: Array of observable values (same shape as j2_j1)
            observable_name: Name of the observable for labeling
            save_name: Filename for saved plot
        """
        logger.info("Generating scaling collapse plot...")
        
        # Extract scaling parameters
        j2_j1_c = scaling_results['j2_j1_c']
        nu = scaling_results['nu']
        x_O = scaling_results['x_O']
        chi_squared = scaling_results.get('chi_squared', 0.0)
        
        # Get uncertainties if available
        if 'nu_uncertainty' in scaling_results:
            nu_unc = scaling_results['nu_uncertainty']
            x_O_unc = scaling_results['x_O_uncertainty']
            j2_j1_c_unc = scaling_results['j2_j1_c_uncertainty']
        else:
            nu_unc = x_O_unc = j2_j1_c_unc = 0.0
        
        # Compute scaled coordinates
        x_scaled = (j2_j1 - j2_j1_c) * np.power(L, 1.0 / nu)
        y_scaled = observable * np.power(L, x_O / nu)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left panel: Raw data
        unique_L = sorted(np.unique(L))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_L)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        for i, L_val in enumerate(unique_L):
            mask = L == L_val
            marker = markers[i % len(markers)]
            ax1.plot(j2_j1[mask], observable[mask], 
                    marker=marker, markersize=8,
                    color=colors[i], label=f'L={L_val}',
                    linewidth=0, alpha=0.7)
        
        # Mark critical point
        ax1.axvline(j2_j1_c, color='red', linestyle='--', linewidth=2,
                   label=f'$j_{{2}}/j_{{1}}^{{c}}$ = {j2_j1_c:.3f}')
        if j2_j1_c_unc > 0:
            ax1.axvspan(j2_j1_c - j2_j1_c_unc, j2_j1_c + j2_j1_c_unc,
                       alpha=0.2, color='red')
        
        ax1.set_xlabel('$J_2/J_1$', fontsize=14)
        ax1.set_ylabel(observable_name, fontsize=14)
        ax1.set_title('Raw Data', fontsize=16)
        ax1.legend(loc='best', framealpha=0.9, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Scaled data (collapse)
        for i, L_val in enumerate(unique_L):
            mask = L == L_val
            marker = markers[i % len(markers)]
            
            # Sort by x_scaled for proper line plotting
            sort_idx = np.argsort(x_scaled[mask])
            x_plot = x_scaled[mask][sort_idx]
            y_plot = y_scaled[mask][sort_idx]
            
            ax2.plot(x_plot, y_plot,
                    marker=marker, markersize=8,
                    color=colors[i], label=f'L={L_val}',
                    linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel(f'$(J_2/J_1 - j_{{2}}/j_{{1}}^{{c}}) L^{{1/\\nu}}$', fontsize=14)
        ax2.set_ylabel(f'{observable_name} $\\times L^{{x_O/\\nu}}$', fontsize=14)
        ax2.set_title('Scaling Collapse', fontsize=16)
        ax2.legend(loc='best', framealpha=0.9, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # Add text box with scaling parameters
        textstr = f'$j_{{2}}/j_{{1}}^{{c}}$ = {j2_j1_c:.4f} ± {j2_j1_c_unc:.4f}\n'
        textstr += f'$\\nu$ = {nu:.4f} ± {nu_unc:.4f}\n'
        textstr += f'$x_O$ = {x_O:.4f} ± {x_O_unc:.4f}\n'
        textstr += f'$\\chi^2$ = {chi_squared:.4e}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.suptitle(f'Finite-Size Scaling Analysis: {observable_name}', 
                    fontsize=18, y=1.00)
        plt.tight_layout()
        
        # Save figure
        save_path = self.scaling_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scaling collapse plot to {save_path}")
        
        plt.close()
    
    def plot_training_curves(self,
                           history: Dict[str, List[float]],
                           lattice_size: Optional[int] = None,
                           save_name: str = 'training_curves.png') -> None:
        """
        Plot Q-VAE training loss curves.
        
        Shows training and validation losses over epochs, including
        total loss, fidelity loss, and KL divergence components.
        
        Args:
            history: Dictionary with keys:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch
                - 'train_fidelity_loss': List of training fidelity losses
                - 'val_fidelity_loss': List of validation fidelity losses
                - 'train_kl_loss': List of training KL losses
                - 'val_kl_loss': List of validation KL losses
            lattice_size: Lattice size for title (optional)
            save_name: Filename for saved plot
        """
        logger.info("Generating training curves plot...")
        
        if not history or 'train_loss' not in history:
            logger.warning("No training history provided")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = np.arange(1, len(history['train_loss']) + 1)
        
        # Plot 1: Total loss
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'o-', label='Training', 
               color='blue', linewidth=2, markersize=4, alpha=0.7)
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 's-', label='Validation',
                   color='red', linewidth=2, markersize=4, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total Loss (ELBO)', fontsize=12)
        ax.set_title('Total Loss', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2: Fidelity loss
        ax = axes[0, 1]
        if 'train_fidelity_loss' in history:
            ax.plot(epochs, history['train_fidelity_loss'], 'o-', label='Training',
                   color='blue', linewidth=2, markersize=4, alpha=0.7)
        if 'val_fidelity_loss' in history:
            ax.plot(epochs, history['val_fidelity_loss'], 's-', label='Validation',
                   color='red', linewidth=2, markersize=4, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Fidelity Loss (1 - F)', fontsize=12)
        ax.set_title('Fidelity Loss', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: KL divergence
        ax = axes[1, 0]
        if 'train_kl_loss' in history:
            ax.plot(epochs, history['train_kl_loss'], 'o-', label='Training',
                   color='blue', linewidth=2, markersize=4, alpha=0.7)
        if 'val_kl_loss' in history:
            ax.plot(epochs, history['val_kl_loss'], 's-', label='Validation',
                   color='red', linewidth=2, markersize=4, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title('KL Divergence', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Reconstruction fidelity (1 - fidelity_loss)
        ax = axes[1, 1]
        if 'train_fidelity_loss' in history:
            train_fidelity = [1.0 - loss for loss in history['train_fidelity_loss']]
            ax.plot(epochs, train_fidelity, 'o-', label='Training',
                   color='blue', linewidth=2, markersize=4, alpha=0.7)
        if 'val_fidelity_loss' in history:
            val_fidelity = [1.0 - loss for loss in history['val_fidelity_loss']]
            ax.plot(epochs, val_fidelity, 's-', label='Validation',
                   color='red', linewidth=2, markersize=4, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reconstruction Fidelity', fontsize=12)
        ax.set_title('Reconstruction Fidelity', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Overall title
        if lattice_size is not None:
            title = f'Q-VAE Training Curves (L={lattice_size})'
        else:
            title = 'Q-VAE Training Curves'
        plt.suptitle(title, fontsize=18, y=1.00)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.training_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        
        plt.close()

    def plot_ensemble_critical_points(self,
                                     detection_results: Dict[str, Tuple[float, float]],
                                     save_name: str = 'ensemble_critical_points.png') -> None:
        """
        Plot comparison of critical points from all detection methods.
        
        Creates a bar chart showing critical point estimates from each method
        with error bars, plus the ensemble estimate.
        
        Args:
            detection_results: Dictionary mapping method_name -> (j2_j1_c, uncertainty)
            save_name: Filename for saved plot
        """
        logger.info("Generating ensemble critical points comparison plot...")
        
        if not detection_results:
            logger.warning("No detection results provided")
            return
        
        # Prepare data
        methods = list(detection_results.keys())
        j2_j1_c_values = [detection_results[m][0] for m in methods]
        uncertainties = [detection_results[m][1] for m in methods]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color ensemble differently
        colors = ['blue' if m != 'ensemble' else 'red' for m in methods]
        
        # Create bar plot
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, j2_j1_c_values, yerr=uncertainties,
                     color=colors, alpha=0.7, capsize=10, edgecolor='black', linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Detection Method', fontsize=14)
        ax.set_ylabel('$J_2/J_1$ Critical Point', fontsize=14)
        ax.set_title('Critical Point Estimates from All Methods', fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val, unc) in enumerate(zip(bars, j2_j1_c_values, uncertainties)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + unc + 0.01,
                   f'{val:.3f}±{unc:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.critical_point_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ensemble critical points plot to {save_path}")
        
        plt.close()
    
    def plot_latent_variance_vs_j2j1(self,
                                    latent_variance_data: Dict[float, float],
                                    critical_points: Optional[List[float]] = None,
                                    save_name: str = 'latent_variance.png') -> None:
        """
        Plot latent variance as a function of j2_j1.
        
        Helper method for detailed visualization of latent variance method.
        
        Args:
            latent_variance_data: Dict mapping j2_j1 -> variance
            critical_points: List of critical point locations to mark
            save_name: Filename for saved plot
        """
        logger.info("Generating latent variance plot...")
        
        if not latent_variance_data:
            logger.warning("No latent variance data provided")
            return
        
        # Prepare data
        j2_j1_vals = np.array(sorted(latent_variance_data.keys()))
        variance_vals = np.array([latent_variance_data[j] for j in j2_j1_vals])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot variance
        ax.plot(j2_j1_vals, variance_vals, 'o-', color='blue',
               linewidth=2, markersize=8, label='Latent Variance')
        
        # Mark critical points if provided
        if critical_points:
            for j2_j1_c in critical_points:
                ax.axvline(j2_j1_c, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('$J_2/J_1$', fontsize=14)
        ax.set_ylabel('Latent Variance', fontsize=14)
        ax.set_title('Latent Variance Across System Sizes', fontsize=16)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.critical_point_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved latent variance plot to {save_path}")
        
        plt.close()
    
    def create_summary_report(self,
                            results: Dict[str, Any],
                            save_name: str = 'analysis_summary.txt') -> None:
        """
        Create a text summary report of all analysis results.
        
        Generates a human-readable summary including:
        - Critical point estimates from all methods
        - Discovered order parameters
        - Scaling exponents
        - Validation results
        
        Args:
            results: Dictionary containing all analysis results
            save_name: Filename for saved report
        """
        logger.info("Generating summary report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("J1-J2 HEISENBERG PROMETHEUS ANALYSIS SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Critical point detection results
        if 'critical_points' in results:
            report_lines.append("CRITICAL POINT DETECTION")
            report_lines.append("-" * 80)
            
            for method, (j2_j1_c, uncertainty) in results['critical_points'].items():
                method_name = method.replace('_', ' ').title()
                report_lines.append(f"{method_name:30s}: j2/j1_c = {j2_j1_c:.4f} ± {uncertainty:.4f}")
            
            report_lines.append("")
        
        # Order parameter discovery results
        if 'order_parameters' in results:
            report_lines.append("DISCOVERED ORDER PARAMETERS")
            report_lines.append("-" * 80)
            
            discovered = results['order_parameters'].get('discovered_order_parameters', {})
            for latent_dim, observable in discovered.items():
                report_lines.append(f"{latent_dim:15s} <-> {observable}")
            
            report_lines.append("")
        
        # Scaling analysis results
        if 'scaling' in results:
            report_lines.append("FINITE-SIZE SCALING ANALYSIS")
            report_lines.append("-" * 80)
            
            scaling = results['scaling']
            report_lines.append(f"Critical Point:              j2/j1_c = {scaling.get('j2_j1_c', 0):.4f}")
            report_lines.append(f"Correlation Length Exponent: nu      = {scaling.get('nu', 0):.4f}")
            report_lines.append(f"Scaling Dimension:           x_O     = {scaling.get('x_O', 0):.4f}")
            report_lines.append(f"Collapse Quality:            chi^2   = {scaling.get('chi_squared', 0):.4e}")
            
            report_lines.append("")
        
        # Validation results
        if 'validation' in results:
            report_lines.append("VALIDATION IN KNOWN PHASES")
            report_lines.append("-" * 80)
            
            validation = results['validation']
            neel_valid = validation.get('neel_phase_valid', False)
            stripe_valid = validation.get('stripe_phase_valid', False)
            
            report_lines.append(f"Néel Phase Validation:   {'PASSED' if neel_valid else 'FAILED'}")
            report_lines.append(f"Stripe Phase Validation: {'PASSED' if stripe_valid else 'FAILED'}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # Write report to file
        report_text = "\n".join(report_lines)
        save_path = self.output_dir / save_name
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Saved summary report to {save_path}")
        
        # Also log to console
        logger.info("\n" + report_text)
