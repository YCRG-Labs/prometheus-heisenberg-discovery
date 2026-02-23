"""
Unit tests for latent space analysis module.

Tests cover:
- Silhouette score computation
- Trajectory arc length computation
- Pairwise distance computation
- Dimensionality reduction (t-SNE)
- Clustering algorithms (k-means, DBSCAN)
"""

import pytest
import numpy as np
from src.latent_space_analysis import (
    LatentSpaceAnalysis,
    ClusteringResult,
    TrajectoryAnalysis,
    DimensionalityReductionResult
)


@pytest.fixture
def latent_analysis():
    """Create LatentSpaceAnalysis instance"""
    return LatentSpaceAnalysis()


@pytest.fixture
def simple_latent_data():
    """Create simple synthetic latent data with clear clusters"""
    np.random.seed(42)
    # Two well-separated clusters
    cluster1 = np.random.randn(20, 4) + np.array([0, 0, 0, 0])
    cluster2 = np.random.randn(20, 4) + np.array([5, 5, 5, 5])
    data = np.vstack([cluster1, cluster2])
    labels = np.array([0] * 20 + [1] * 20)
    return data, labels


@pytest.fixture
def trajectory_data():
    """Create smooth trajectory data"""
    t = np.linspace(0, 2*np.pi, 50)
    # Parametric curve in 3D
    trajectory = np.column_stack([
        np.cos(t),
        np.sin(t),
        t / (2*np.pi)
    ])
    return trajectory, t


class TestSilhouetteScore:
    """Test silhouette score computation"""
    
    def test_perfect_separation(self, latent_analysis, simple_latent_data):
        """Test silhouette score with well-separated clusters"""
        data, labels = simple_latent_data
        score = latent_analysis.compute_silhouette_score(data, labels)
        
        # Well-separated clusters should have high silhouette score
        assert 0.5 < score <= 1.0
        
    def test_single_cluster(self, latent_analysis):
        """Test silhouette score with single cluster returns 0"""
        data = np.random.randn(10, 3)
        labels = np.zeros(10, dtype=int)
        
        score = latent_analysis.compute_silhouette_score(data, labels)
        assert score == 0.0
        
    def test_overlapping_clusters(self, latent_analysis):
        """Test silhouette score with overlapping clusters"""
        np.random.seed(42)
        # Two overlapping clusters
        cluster1 = np.random.randn(20, 3)
        cluster2 = np.random.randn(20, 3) + 0.5  # Small separation
        data = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 20 + [1] * 20)
        
        score = latent_analysis.compute_silhouette_score(data, labels)
        # Overlapping clusters should have lower score
        assert -1.0 <= score <= 1.0
        
    def test_different_metrics(self, latent_analysis, simple_latent_data):
        """Test silhouette score with different distance metrics"""
        data, labels = simple_latent_data
        
        euclidean_score = latent_analysis.compute_silhouette_score(
            data, labels, metric='euclidean'
        )
        manhattan_score = latent_analysis.compute_silhouette_score(
            data, labels, metric='manhattan'
        )
        
        # Both should be valid scores
        assert -1.0 <= euclidean_score <= 1.0
        assert -1.0 <= manhattan_score <= 1.0


class TestTrajectoryArcLength:
    """Test trajectory arc length computation"""
    
    def test_smooth_trajectory(self, latent_analysis, trajectory_data):
        """Test arc length for smooth trajectory"""
        trajectory, t = trajectory_data
        result = latent_analysis.compute_trajectory_arc_length(trajectory, t)
        
        assert isinstance(result, TrajectoryAnalysis)
        assert result.arc_length > 0
        assert 0 <= result.smoothness <= 1.0
        assert len(result.discontinuities) == 0  # Smooth trajectory
        
    def test_discontinuous_trajectory(self, latent_analysis):
        """Test arc length with discontinuity"""
        # Create trajectory with a jump
        trajectory = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [10, 10],  # Large jump
            [11, 11],
            [12, 12]
        ])
        
        result = latent_analysis.compute_trajectory_arc_length(trajectory)
        
        assert result.arc_length > 0
        assert len(result.discontinuities) > 0  # Should detect jump
        assert 2 in result.discontinuities or 3 in result.discontinuities
        
    def test_single_point_trajectory(self, latent_analysis):
        """Test arc length with single point"""
        trajectory = np.array([[1, 2, 3]])
        result = latent_analysis.compute_trajectory_arc_length(trajectory)
        
        assert result.arc_length == 0.0
        assert result.smoothness == 1.0
        assert len(result.discontinuities) == 0
        
    def test_two_point_trajectory(self, latent_analysis):
        """Test arc length with two points"""
        trajectory = np.array([[0, 0], [3, 4]])
        result = latent_analysis.compute_trajectory_arc_length(trajectory)
        
        # Distance should be 5 (3-4-5 triangle)
        assert abs(result.arc_length - 5.0) < 1e-10
        assert result.smoothness == 1.0


class TestPairwiseDistances:
    """Test pairwise distance computation"""
    
    def test_distance_matrix_shape(self, latent_analysis):
        """Test pairwise distance matrix has correct shape"""
        data = np.random.randn(10, 5)
        distances = latent_analysis.compute_pairwise_distances(data)
        
        assert distances.shape == (10, 10)
        
    def test_distance_symmetry(self, latent_analysis):
        """Test distance matrix is symmetric"""
        data = np.random.randn(5, 3)
        distances = latent_analysis.compute_pairwise_distances(data)
        
        assert np.allclose(distances, distances.T)
        
    def test_diagonal_zeros(self, latent_analysis):
        """Test diagonal elements are zero (distance to self)"""
        data = np.random.randn(5, 3)
        distances = latent_analysis.compute_pairwise_distances(data)
        
        assert np.allclose(np.diag(distances), 0)
        
    def test_non_negative_distances(self, latent_analysis):
        """Test all distances are non-negative"""
        data = np.random.randn(10, 4)
        distances = latent_analysis.compute_pairwise_distances(data)
        
        assert np.all(distances >= 0)
        
    def test_known_distances(self, latent_analysis):
        """Test distances with known values"""
        # Simple 2D points
        data = np.array([
            [0, 0],
            [3, 4],
            [0, 1]
        ])
        distances = latent_analysis.compute_pairwise_distances(data)
        
        # Distance from [0,0] to [3,4] should be 5
        assert abs(distances[0, 1] - 5.0) < 1e-10
        # Distance from [0,0] to [0,1] should be 1
        assert abs(distances[0, 2] - 1.0) < 1e-10


class TestDimensionalityReduction:
    """Test dimensionality reduction methods"""
    
    def test_tsne_reduction(self, latent_analysis, simple_latent_data):
        """Test t-SNE dimensionality reduction"""
        data, _ = simple_latent_data
        result = latent_analysis.reduce_dimensionality(
            data, method='tsne', n_components=2
        )
        
        assert isinstance(result, DimensionalityReductionResult)
        assert result.embedding.shape == (len(data), 2)
        assert result.method == 'tsne'
        
    def test_tsne_3d(self, latent_analysis):
        """Test t-SNE reduction to 3D"""
        data = np.random.randn(30, 8)
        result = latent_analysis.reduce_dimensionality(
            data, method='tsne', n_components=3
        )
        
        assert result.embedding.shape == (30, 3)
        
    def test_invalid_method(self, latent_analysis):
        """Test invalid dimensionality reduction method"""
        data = np.random.randn(20, 5)
        
        with pytest.raises(ValueError, match="Unknown method"):
            latent_analysis.reduce_dimensionality(data, method='invalid')
            
    def test_tsne_reproducibility(self, latent_analysis):
        """Test t-SNE with fixed random seed is reproducible"""
        data = np.random.randn(20, 5)
        
        result1 = latent_analysis.reduce_dimensionality(
            data, method='tsne', random_state=42
        )
        result2 = latent_analysis.reduce_dimensionality(
            data, method='tsne', random_state=42
        )
        
        assert np.allclose(result1.embedding, result2.embedding)


class TestKMeansClustering:
    """Test k-means clustering"""
    
    def test_kmeans_basic(self, latent_analysis, simple_latent_data):
        """Test k-means clustering with clear clusters"""
        data, true_labels = simple_latent_data
        result = latent_analysis.cluster_kmeans(data, n_clusters=2)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.labels) == len(data)
        assert result.n_clusters == 2
        assert result.algorithm == 'kmeans'
        assert 0.5 < result.silhouette_score <= 1.0  # Good separation
        
    def test_kmeans_three_clusters(self, latent_analysis):
        """Test k-means with three clusters"""
        np.random.seed(42)
        cluster1 = np.random.randn(15, 3) + [0, 0, 0]
        cluster2 = np.random.randn(15, 3) + [5, 0, 0]
        cluster3 = np.random.randn(15, 3) + [0, 5, 0]
        data = np.vstack([cluster1, cluster2, cluster3])
        
        result = latent_analysis.cluster_kmeans(data, n_clusters=3)
        
        assert result.n_clusters == 3
        assert len(np.unique(result.labels)) == 3
        
    def test_kmeans_reproducibility(self, latent_analysis):
        """Test k-means with fixed random seed is reproducible"""
        data = np.random.randn(30, 4)
        
        result1 = latent_analysis.cluster_kmeans(data, n_clusters=2, random_state=42)
        result2 = latent_analysis.cluster_kmeans(data, n_clusters=2, random_state=42)
        
        assert np.array_equal(result1.labels, result2.labels)


class TestDBSCANClustering:
    """Test DBSCAN clustering"""
    
    def test_dbscan_basic(self, latent_analysis, simple_latent_data):
        """Test DBSCAN clustering with clear clusters"""
        data, _ = simple_latent_data
        result = latent_analysis.cluster_dbscan(data, eps=1.0, min_samples=3)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.labels) == len(data)
        assert result.algorithm == 'dbscan'
        assert result.n_clusters >= 1
        
    def test_dbscan_noise_detection(self, latent_analysis):
        """Test DBSCAN detects noise points"""
        np.random.seed(42)
        # Dense cluster + scattered noise
        cluster = np.random.randn(20, 2) * 0.3
        noise = np.random.randn(5, 2) * 5
        data = np.vstack([cluster, noise])
        
        result = latent_analysis.cluster_dbscan(data, eps=0.5, min_samples=3)
        
        # Should have noise points (label -1)
        assert -1 in result.labels
        
    def test_dbscan_single_cluster(self, latent_analysis):
        """Test DBSCAN with single dense cluster"""
        np.random.seed(42)
        data = np.random.randn(30, 3) * 0.5
        
        result = latent_analysis.cluster_dbscan(data, eps=1.0, min_samples=3)
        
        # Should find at least one cluster
        assert result.n_clusters >= 1


class TestAnalyzeLatentStructure:
    """Test comprehensive latent structure analysis"""
    
    def test_analyze_single_lattice_size(self, latent_analysis):
        """Test comprehensive analysis for single lattice size"""
        # Create synthetic data
        np.random.seed(42)
        latent_reps = {}
        j2_j1_values = np.linspace(0.3, 0.7, 20)
        
        for j2_j1 in j2_j1_values:
            z = np.random.randn(4) + j2_j1  # Latent varies with parameter
            latent_reps[(j2_j1, 4)] = z
            
        results = latent_analysis.analyze_latent_structure(latent_reps, L=4)
        
        assert 4 in results
        assert 'trajectory' in results[4]
        assert 'distances' in results[4]
        assert 'clustering' in results[4]
        assert 'j2_j1_values' in results[4]
        
    def test_analyze_multiple_lattice_sizes(self, latent_analysis):
        """Test analysis with multiple lattice sizes"""
        np.random.seed(42)
        latent_reps = {}
        
        for L in [4, 5, 6]:
            for j2_j1 in np.linspace(0.3, 0.7, 10):
                z = np.random.randn(4)
                latent_reps[(j2_j1, L)] = z
                
        results = latent_analysis.analyze_latent_structure(latent_reps)
        
        assert len(results) == 3
        assert 4 in results
        assert 5 in results
        assert 6 in results
        
    def test_trajectory_ordering(self, latent_analysis):
        """Test that trajectory is ordered by j2_j1"""
        np.random.seed(42)
        latent_reps = {}
        j2_j1_values = [0.5, 0.3, 0.7, 0.4, 0.6]  # Unordered
        
        for j2_j1 in j2_j1_values:
            z = np.array([j2_j1, 0, 0, 0])  # Encode j2_j1 in first component
            latent_reps[(j2_j1, 4)] = z
            
        results = latent_analysis.analyze_latent_structure(latent_reps, L=4)
        
        # Check that j2_j1 values are sorted
        j2_j1_result = results[4]['j2_j1_values']
        assert np.all(j2_j1_result[:-1] <= j2_j1_result[1:])
