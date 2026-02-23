"""
Latent Space Analysis Module

This module implements latent space structure analysis functions including:
- Silhouette score computation for cluster quality
- Trajectory arc length computation for smoothness analysis
- Pairwise distance computation
- Dimensionality reduction (t-SNE/UMAP) for visualization
- Clustering algorithms (k-means, DBSCAN)

Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 7.7, 7.8
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

from src.logging_config import LoggerMixin


@dataclass
class ClusteringResult:
    """Result of clustering analysis"""
    labels: np.ndarray
    silhouette_score: float
    n_clusters: int
    algorithm: str
    parameters: Dict[str, Any]


@dataclass
class TrajectoryAnalysis:
    """Result of trajectory analysis"""
    arc_length: float
    smoothness: float
    discontinuities: List[int]  # Indices where large jumps occur
    j2_j1_values: np.ndarray


@dataclass
class DimensionalityReductionResult:
    """Result of dimensionality reduction"""
    embedding: np.ndarray
    method: str
    parameters: Dict[str, Any]


class LatentSpaceAnalysis(LoggerMixin):
    """
    Latent space structure analysis.
    
    Provides methods for:
    - Computing silhouette scores for cluster quality assessment
    - Computing trajectory arc lengths for smoothness analysis
    - Computing pairwise distances between latent representations
    - Dimensionality reduction for visualization (t-SNE, UMAP)
    - Clustering analysis (k-means, DBSCAN)
    """
    
    def __init__(self, config=None):
        """
        Initialize latent space analysis.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        
    def compute_silhouette_score(
        self,
        latent_reps: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean'
    ) -> float:
        """
        Compute silhouette score to quantify phase separation quality.
        
        The silhouette score measures how well-separated clusters are.
        Score ranges from -1 to 1:
        - 1: Perfect separation
        - 0: Overlapping clusters
        - -1: Incorrect clustering
        
        Args:
            latent_reps: Array of shape (n_samples, latent_dim)
            labels: Cluster labels for each sample
            metric: Distance metric (default: 'euclidean')
            
        Returns:
            Silhouette score in range [-1, 1]
            
        Requirements: 7.1
        """
        if len(np.unique(labels)) < 2:
            self.logger.warning("Less than 2 clusters, silhouette score undefined")
            return 0.0
            
        score = silhouette_score(latent_reps, labels, metric=metric)
        self.logger.debug(f"Silhouette score: {score:.4f}")
        return float(score)
        
    def compute_trajectory_arc_length(
        self,
        latent_trajectory: np.ndarray,
        j2_j1_values: Optional[np.ndarray] = None
    ) -> TrajectoryAnalysis:
        """
        Compute trajectory arc length to measure smoothness of latent evolution.
        
        Arc length = Σ ||z(i+1) - z(i)||
        
        Large arc length indicates rapid changes in latent space.
        Discontinuities (large jumps) may indicate phase transitions.
        
        Args:
            latent_trajectory: Array of shape (n_points, latent_dim) ordered by parameter
            j2_j1_values: Optional array of j2_j1 values corresponding to trajectory points
            
        Returns:
            TrajectoryAnalysis with arc length, smoothness, and discontinuities
            
        Requirements: 7.2, 7.3
        """
        if len(latent_trajectory) < 2:
            return TrajectoryAnalysis(
                arc_length=0.0,
                smoothness=1.0,
                discontinuities=[],
                j2_j1_values=j2_j1_values if j2_j1_values is not None else np.array([])
            )
        
        # Compute distances between consecutive points
        diffs = np.diff(latent_trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        # Total arc length
        arc_length = float(np.sum(distances))
        
        # Identify discontinuities (jumps > 2 * median distance)
        median_dist = np.median(distances)
        threshold = 2.0 * median_dist
        discontinuities = np.where(distances > threshold)[0].tolist()
        
        # Smoothness metric: inverse of variance in step sizes
        # High variance = low smoothness
        if len(distances) > 1:
            step_variance = np.var(distances)
            smoothness = float(1.0 / (1.0 + step_variance))
        else:
            smoothness = 1.0
            
        self.logger.debug(
            f"Arc length: {arc_length:.4f}, "
            f"Smoothness: {smoothness:.4f}, "
            f"Discontinuities: {len(discontinuities)}"
        )
        
        return TrajectoryAnalysis(
            arc_length=arc_length,
            smoothness=smoothness,
            discontinuities=discontinuities,
            j2_j1_values=j2_j1_values if j2_j1_values is not None else np.array([])
        )
        
    def compute_pairwise_distances(
        self,
        latent_reps: np.ndarray,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """
        Compute pairwise distances between all latent representations.
        
        Args:
            latent_reps: Array of shape (n_samples, latent_dim)
            metric: Distance metric (default: 'euclidean')
            
        Returns:
            Distance matrix of shape (n_samples, n_samples)
            
        Requirements: 7.6
        """
        distances = pairwise_distances(latent_reps, metric=metric)
        self.logger.debug(
            f"Computed pairwise distances: shape {distances.shape}, "
            f"mean={np.mean(distances):.4f}, std={np.std(distances):.4f}"
        )
        return distances
        
    def reduce_dimensionality(
        self,
        latent_reps: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        **kwargs
    ) -> DimensionalityReductionResult:
        """
        Perform dimensionality reduction for visualization.
        
        Supports t-SNE and UMAP for reducing high-dimensional latent space
        to 2D or 3D for visualization.
        
        Args:
            latent_reps: Array of shape (n_samples, latent_dim)
            method: 'tsne' or 'umap'
            n_components: Target dimensionality (2 or 3)
            **kwargs: Additional parameters for the reduction method
            
        Returns:
            DimensionalityReductionResult with embedding and metadata
            
        Requirements: 7.4
        """
        method = method.lower()
        
        if method == 'tsne':
            # Default t-SNE parameters
            tsne_params = {
                'n_components': n_components,
                'perplexity': min(30, len(latent_reps) - 1),
                'random_state': kwargs.get('random_state', 42),
                'n_iter': kwargs.get('n_iter', 1000)
            }
            tsne_params.update(kwargs)
            
            self.logger.info(f"Running t-SNE with parameters: {tsne_params}")
            reducer = TSNE(**tsne_params)
            embedding = reducer.fit_transform(latent_reps)
            
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "UMAP not available. Install with: pip install umap-learn"
                )
            
            # Default UMAP parameters
            umap_params = {
                'n_components': n_components,
                'n_neighbors': min(15, len(latent_reps) - 1),
                'min_dist': 0.1,
                'random_state': kwargs.get('random_state', 42)
            }
            umap_params.update(kwargs)
            
            self.logger.info(f"Running UMAP with parameters: {umap_params}")
            reducer = umap.UMAP(**umap_params)
            embedding = reducer.fit_transform(latent_reps)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")
            
        self.logger.info(
            f"Dimensionality reduction complete: {latent_reps.shape} -> {embedding.shape}"
        )
        
        return DimensionalityReductionResult(
            embedding=embedding,
            method=method,
            parameters=tsne_params if method == 'tsne' else umap_params
        )
        
    def cluster_kmeans(
        self,
        latent_reps: np.ndarray,
        n_clusters: int,
        **kwargs
    ) -> ClusteringResult:
        """
        Perform k-means clustering on latent representations.
        
        Args:
            latent_reps: Array of shape (n_samples, latent_dim)
            n_clusters: Number of clusters
            **kwargs: Additional parameters for KMeans
            
        Returns:
            ClusteringResult with labels and quality metrics
            
        Requirements: 7.7
        """
        kmeans_params = {
            'n_clusters': n_clusters,
            'random_state': kwargs.get('random_state', 42),
            'n_init': kwargs.get('n_init', 10)
        }
        kmeans_params.update(kwargs)
        
        self.logger.info(f"Running k-means with {n_clusters} clusters")
        kmeans = KMeans(**kmeans_params)
        labels = kmeans.fit_predict(latent_reps)
        
        # Compute silhouette score
        sil_score = self.compute_silhouette_score(latent_reps, labels)
        
        self.logger.info(
            f"K-means clustering complete: {n_clusters} clusters, "
            f"silhouette score: {sil_score:.4f}"
        )
        
        return ClusteringResult(
            labels=labels,
            silhouette_score=sil_score,
            n_clusters=n_clusters,
            algorithm='kmeans',
            parameters=kmeans_params
        )
        
    def cluster_dbscan(
        self,
        latent_reps: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """
        Perform DBSCAN clustering on latent representations.
        
        DBSCAN is density-based and can discover clusters of arbitrary shape.
        It also identifies noise points (label = -1).
        
        Args:
            latent_reps: Array of shape (n_samples, latent_dim)
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood to form core point
            **kwargs: Additional parameters for DBSCAN
            
        Returns:
            ClusteringResult with labels and quality metrics
            
        Requirements: 7.7
        """
        dbscan_params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': kwargs.get('metric', 'euclidean')
        }
        dbscan_params.update(kwargs)
        
        self.logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        dbscan = DBSCAN(**dbscan_params)
        labels = dbscan.fit_predict(latent_reps)
        
        # Count clusters (excluding noise points with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Compute silhouette score (only if we have at least 2 clusters)
        if n_clusters >= 2:
            # Exclude noise points for silhouette calculation
            mask = labels != -1
            if np.sum(mask) > 0:
                sil_score = self.compute_silhouette_score(
                    latent_reps[mask], 
                    labels[mask]
                )
            else:
                sil_score = 0.0
        else:
            sil_score = 0.0
            
        self.logger.info(
            f"DBSCAN clustering complete: {n_clusters} clusters, "
            f"{n_noise} noise points, silhouette score: {sil_score:.4f}"
        )
        
        return ClusteringResult(
            labels=labels,
            silhouette_score=sil_score,
            n_clusters=n_clusters,
            algorithm='dbscan',
            parameters=dbscan_params
        )
        
    def analyze_latent_structure(
        self,
        latent_reps: Dict[Tuple[float, int], np.ndarray],
        L: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive latent space structure analysis.
        
        Performs multiple analyses:
        - Trajectory analysis for each lattice size
        - Clustering analysis
        - Dimensionality reduction
        - Pairwise distance analysis
        
        Args:
            latent_reps: Dictionary mapping (j2_j1, L) -> latent vector
            L: Optional lattice size to analyze (if None, analyze all)
            
        Returns:
            Dictionary with analysis results
            
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 7.7, 7.8
        """
        results = {}
        
        # Organize data by lattice size
        data_by_L = {}
        for (j2_j1, lattice_size), z in latent_reps.items():
            if L is not None and lattice_size != L:
                continue
            if lattice_size not in data_by_L:
                data_by_L[lattice_size] = []
            data_by_L[lattice_size].append((j2_j1, z))
            
        # Analyze each lattice size
        for lattice_size, data in data_by_L.items():
            self.logger.info(f"Analyzing latent structure for L={lattice_size}")
            
            # Sort by j2_j1 for trajectory analysis
            data.sort(key=lambda x: x[0])
            j2_j1_values = np.array([j2_j1 for j2_j1, _ in data])
            latent_array = np.array([z for _, z in data])
            
            # Trajectory analysis
            trajectory = self.compute_trajectory_arc_length(
                latent_array, 
                j2_j1_values
            )
            
            # Pairwise distances
            distances = self.compute_pairwise_distances(latent_array)
            
            # Try clustering with different numbers of clusters
            clustering_results = []
            for n_clusters in [2, 3]:
                if len(latent_array) >= n_clusters:
                    cluster_result = self.cluster_kmeans(latent_array, n_clusters)
                    clustering_results.append(cluster_result)
                    
            # Dimensionality reduction for visualization
            if len(latent_array) > 3:
                try:
                    tsne_result = self.reduce_dimensionality(
                        latent_array, 
                        method='tsne',
                        n_components=2
                    )
                except Exception as e:
                    self.logger.warning(f"t-SNE failed: {e}")
                    tsne_result = None
            else:
                tsne_result = None
                
            results[lattice_size] = {
                'trajectory': trajectory,
                'distances': distances,
                'clustering': clustering_results,
                'dimensionality_reduction': tsne_result,
                'j2_j1_values': j2_j1_values,
                'latent_array': latent_array
            }
            
        return results
