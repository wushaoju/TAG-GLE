#!/usr/bin/env python3
"""
NIfTI File Loading and 3D Sharpness Visualization
Complete script with all necessary imports
"""

# Standard library imports
import os
import warnings
from pathlib import Path

# Essential numerical and scientific computing
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# Machine learning utilities
from sklearn.utils import shuffle

# Data analysis and visualization
import pandas as pd
import scipy.io
# Optional imports with availability checks
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. 3D interactive visualizations will be disabled.")
    print("Install with: pip install plotly")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress bar
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: Seaborn not available. Some plotting features will be limited.")

# Suppress warnings if needed
warnings.filterwarnings('ignore', category=UserWarning)


class SimpleSharpnessVisualizer:
    """Simplified version for loading examples with sharpness computation"""
    
    def __init__(self, patch_size=5, grid_spacing=8, num_patches=1000):
        self.patch_size = patch_size
        self.grid_spacing = grid_spacing
        self.num_patches = num_patches
    
    def load_and_validate_nifti(self, file_path):
        """Load and validate NIfTI file with comprehensive error handling."""
        try:
            print(f"Loading NIfTI file: {file_path}")
            
            # Load the NIfTI file
            nii_img = nib.load(str(file_path))
            image_data = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header
            
            # Print basic information
            print(f"? Successfully loaded: {Path(file_path).name}")
            print(f"  Image shape: {image_data.shape}")
            print(f"  Data type: {image_data.dtype}")
            print(f"  Value range: [{np.min(image_data):.4f}, {np.max(image_data):.4f}]")
            print(f"  Voxel size: {header.get_zooms()[:3]} mm")
            print(f"  Image orientation: {nib.aff2axcodes(affine)}")
            
            # Validation checks
            if image_data.ndim != 3:
                if image_data.ndim == 4:
                    print(f"  Note: 4D image detected, using first volume")
                    image_data = image_data[:, :, :, 0]
                else:
                    raise ValueError(f"Expected 3D image, got {image_data.ndim}D")
            
            if not np.isfinite(image_data).all():
                print("  Warning: Image contains NaN or infinite values")
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check for empty image
            if np.all(image_data == 0):
                raise ValueError("Image appears to be empty (all zeros)")
            
            return image_data, affine, header
            
        except Exception as e:
            print(f"? Error loading {file_path}: {e}")
            return None, None, None
    
    def create_threshold_mask(self, image, threshold=0.15):
        """Create binary mask based on intensity threshold."""
        # Normalize image to [0, 1] range
        img_min, img_max = np.min(image), np.max(image)
        if img_max > img_min:
            normalized_image = (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Create mask
        mask = (normalized_image > threshold).astype(np.uint8)
        
        # Print mask statistics
        total_voxels = np.prod(image.shape)
        masked_voxels = np.sum(mask)
        mask_percentage = (masked_voxels / total_voxels) * 100
        
        print(f"Mask Statistics:")
        print(f"  Threshold: {threshold}")
        print(f"  Coverage: {mask_percentage:.1f}% ({masked_voxels:,} voxels)")
        
        return mask
    
    def compute_mean_sharpness(self, image, mask=None, method='normalized_std'):
        """
        Compute mean sharpness across the entire image or masked region.
        
        Parameters:
        -----------
        image : numpy.ndarray
            3D image data
        mask : numpy.ndarray, optional
            Binary mask to restrict analysis
        method : str
            Sharpness metric ('normalized_std', 'gradient', 'laplacian', 'all')
            
        Returns:
        --------
        dict or float
            Mean sharpness value(s)
        """
        print(f"Computing mean sharpness using {method} method...")
        
        # Get valid coordinates for patch sampling
        half_patch = self.patch_size // 2
        h, w, d = image.shape
        
        # Create list of valid coordinates
        valid_coords = []
        for i in range(half_patch, h - half_patch, self.grid_spacing):
            for j in range(half_patch, w - half_patch, self.grid_spacing):
                for k in range(half_patch, d - half_patch, self.grid_spacing):
                    if mask is None or mask[i, j, k] > 0:
                        valid_coords.append((i, j, k))
        
        if len(valid_coords) == 0:
            print("No valid coordinates found for sharpness computation!")
            return 0.0 if method != 'all' else {'normalized_std': 0.0, 'gradient': 0.0, 'laplacian': 0.0}
        
        # Sample coordinates if too many
        if len(valid_coords) > self.num_patches:
            sampled_coords = shuffle(valid_coords, n_samples=self.num_patches, random_state=42)
        else:
            sampled_coords = valid_coords
        
        print(f"Sampling {len(sampled_coords)} patches from {len(valid_coords)} valid locations...")
        
        # Compute sharpness values
        if method == 'all':
            sharpness_values = {'normalized_std': [], 'gradient': [], 'laplacian': []}
        else:
            sharpness_values = []
        
        for coord in sampled_coords:
            i, j, k = coord
            
            # Extract patch
            patch = image[i-half_patch:i+half_patch+1,
                         j-half_patch:j+half_patch+1,
                         k-half_patch:k+half_patch+1]
            
            if patch.size > 0:
                if method == 'normalized_std':
                    sharpness = self._compute_normalized_std(patch)
                    sharpness_values.append(sharpness)
                elif method == 'gradient':
                    sharpness = self._compute_gradient_magnitude(patch)
                    sharpness_values.append(sharpness)
                elif method == 'laplacian':
                    sharpness = self._compute_laplacian(patch)
                    sharpness_values.append(sharpness)
                elif method == 'all':
                    sharpness_values['normalized_std'].append(self._compute_normalized_std(patch))
                    sharpness_values['gradient'].append(self._compute_gradient_magnitude(patch))
                    sharpness_values['laplacian'].append(self._compute_laplacian(patch))
        
        # Calculate mean values
        if method == 'all':
            results = {}
            for metric_name, values in sharpness_values.items():
                if len(values) > 0:
                    results[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'samples': len(values)
                    }
                else:
                    results[metric_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'samples': 0
                    }
            return results
        else:
            if len(sharpness_values) > 0:
                mean_sharpness = np.mean(sharpness_values)
                std_sharpness = np.std(sharpness_values)
                
                print(f"Mean {method} sharpness: {mean_sharpness:.4f} � {std_sharpness:.4f}")
                print(f"Range: [{np.min(sharpness_values):.4f}, {np.max(sharpness_values):.4f}]")
                print(f"Based on {len(sharpness_values)} patches")
                
                return mean_sharpness
            else:
                return 0.0
    
    def _compute_normalized_std(self, patch):
        """Compute normalized standard deviation sharpness."""
        if patch.size == 0:
            return 0.0
        
        patch_std = np.std(patch)
        patch_mean = np.mean(patch)
        
        if patch_mean > 1e-6:
            return patch_std / patch_mean
        else:
            return 0.0
    
    def _compute_gradient_magnitude(self, patch):
        """Compute gradient magnitude sharpness."""
        if patch.size == 0:
            return 0.0
        
        grad_x = ndimage.sobel(patch, axis=0)
        grad_y = ndimage.sobel(patch, axis=1)
        grad_z = ndimage.sobel(patch, axis=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return np.mean(grad_magnitude)
    
    def _compute_laplacian(self, patch):
        """Compute Laplacian sharpness."""
        if patch.size == 0:
            return 0.0
        
        laplacian = ndimage.laplace(patch)
        return np.mean(np.abs(laplacian))
    
    def quick_visualization(self, image, mask=None, title="NIfTI Visualization"):
        """Create quick 2D slice visualization."""
        # Get middle slices
        mid_slices = [s // 2 for s in image.shape]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image slices
        axes[0, 0].imshow(image[mid_slices[0], :, :], cmap='gray')
        axes[0, 0].set_title(f'Sagittal (X={mid_slices[0]})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(image[:, mid_slices[1], :], cmap='gray')
        axes[0, 1].set_title(f'Coronal (Y={mid_slices[1]})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(image[:, :, mid_slices[2]], cmap='gray')
        axes[0, 2].set_title(f'Axial (Z={mid_slices[2]})')
        axes[0, 2].axis('off')
        
        # Mask overlay (if provided)
        if mask is not None:
            axes[1, 0].imshow(image[mid_slices[0], :, :], cmap='gray', alpha=0.7)
            axes[1, 0].imshow(mask[mid_slices[0], :, :], cmap='Reds', alpha=0.3)
            axes[1, 0].set_title('Sagittal + Mask')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(image[:, mid_slices[1], :], cmap='gray', alpha=0.7)
            axes[1, 1].imshow(mask[:, mid_slices[1], :], cmap='Reds', alpha=0.3)
            axes[1, 1].set_title('Coronal + Mask')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(image[:, :, mid_slices[2]], cmap='gray', alpha=0.7)
            axes[1, 2].imshow(mask[:, :, mid_slices[2]], cmap='Reds', alpha=0.3)
            axes[1, 2].set_title('Axial + Mask')
            axes[1, 2].axis('off')
        else:
            # Show intensity histograms
            axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7)
            axes[1, 0].set_title('Intensity Histogram')
            axes[1, 0].set_xlabel('Intensity')
            axes[1, 0].set_ylabel('Frequency')
            
            # Show image statistics
            axes[1, 1].text(0.1, 0.8, f"Shape: {image.shape}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.7, f"Min: {np.min(image):.4f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f"Max: {np.max(image):.4f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.5, f"Mean: {np.mean(image):.4f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f"Std: {np.std(image):.4f}", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Image Statistics')
            axes[1, 1].axis('off')
            
            axes[1, 2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def comprehensive_analysis(self, image, mask, title="Comprehensive Analysis"):
        """Perform comprehensive analysis including sharpness metrics."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS: {title}")
        print(f"{'='*60}")
        
        # Basic statistics
        basic_stats = {
            'shape': image.shape,
            'total_voxels': np.prod(image.shape),
            'value_range': [float(np.min(image)), float(np.max(image))],
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image))
        }
        
        # Mask statistics
        mask_stats = {
            'masked_voxels': int(np.sum(mask)),
            'mask_coverage_percent': float((np.sum(mask) / np.prod(image.shape)) * 100)
        }
        
        # Compute sharpness metrics
        sharpness_metrics = self.compute_mean_sharpness(image, mask, method='all')
        
        # Visualization
        self.quick_visualization(image, mask, title)
        
        # Compile results
        results = {
            'title': title,
            'basic_stats': basic_stats,
            'mask_stats': mask_stats,
            'sharpness_metrics': sharpness_metrics
        }
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"  Shape: {basic_stats['shape']}")
        print(f"  Mask coverage: {mask_stats['mask_coverage_percent']:.1f}%")
        print(f"  Normalized Std Sharpness: {sharpness_metrics['normalized_std']['mean']:.4f}")
        print(f"  Gradient Sharpness: {sharpness_metrics['gradient']['mean']:.4f}")
        print(f"  Laplacian Sharpness: {sharpness_metrics['laplacian']['mean']:.4f}")
        
        return results


def example_1_single_file_loading():
    """
    Example 1: Load a single NIfTI file and create basic visualization with sharpness analysis
    """
    print("="*70)
    print("EXAMPLE 1: SINGLE NIFTI FILE LOADING WITH SHARPNESS ANALYSIS")
    print("="*70)
    
    # REPLACE THIS PATH WITH YOUR ACTUAL NIFTI FILE
    nifti_file_path = "/path/to/your/brain.nii.gz"
    
    # For demonstration, let's check if the file exists
    if not Path(nifti_file_path).exists():
        print(f"File not found: {nifti_file_path}")
        print("Please update the file path in the code to point to your actual NIfTI file.")
        print("\nExample file paths:")
        print("  - Linux/Mac: '/home/user/data/brain_atlas.nii.gz'")
        print("  - Windows: 'C:\\Data\\brain_atlas.nii.gz'")
        print("  - Relative: './data/brain_atlas.nii.gz'")
        
        # Create synthetic data for demo
        print("\nCreating synthetic data for demonstration...")
        image_data, affine = create_synthetic_brain_data()
        
        # Initialize visualizer
        visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=6, num_patches=500)
        
        # Create threshold mask
        mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
        
        # Comprehensive analysis with sharpness
        analysis_results = visualizer.comprehensive_analysis(image_data, mask, title="Synthetic Brain Data")
        
        return image_data, affine, None, mask, analysis_results
    
    # Initialize visualizer
    visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=6, num_patches=1000)
    
    # Load the NIfTI file
    image_data, affine, header = visualizer.load_and_validate_nifti(nifti_file_path)
    
    if image_data is not None:
        # Create threshold mask
        mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
        
        # Comprehensive analysis with sharpness computation
        analysis_results = visualizer.comprehensive_analysis(
            image_data, mask, title=f"Loaded: {Path(nifti_file_path).name}"
        )
        
        return image_data, affine, header, mask, analysis_results
    
    return None


def example_2_folder_browsing():
    """
    Example 2: Browse a folder and select a NIfTI file to load
    """
    print("="*70)
    print("EXAMPLE 2: FOLDER BROWSING FOR NIFTI FILES")
    print("="*70)
    
    # REPLACE THIS PATH WITH YOUR ACTUAL FOLDER CONTAINING NIFTI FILES
    folder_path = "/path/to/your/nifti/folder"
    
    # Check if folder exists
    if not Path(folder_path).exists():
        print(f"Folder not found: {folder_path}")
        print("Please update the folder path in the code.")
        return None
    
    # Find all NIfTI files in the folder
    nifti_files = []
    for pattern in ['*.nii', '*.nii.gz']:
        nifti_files.extend(list(Path(folder_path).glob(pattern)))
        nifti_files.extend(list(Path(folder_path).rglob(pattern)))  # Recursive search
    
    if len(nifti_files) == 0:
        print(f"No NIfTI files found in {folder_path}")
        return None
    
    # Remove duplicates and sort
    nifti_files = sorted(list(set(nifti_files)))
    
    print(f"Found {len(nifti_files)} NIfTI files:")
    for i, file_path in enumerate(nifti_files[:10]):  # Show first 10
        print(f"  {i+1}. {file_path.name}")
    
    if len(nifti_files) > 10:
        print(f"  ... and {len(nifti_files) - 10} more files")
    
    # For automatic processing, select the first file
    # In interactive use, you could ask user to select
    selected_file = nifti_files[0]
    print(f"\nAutomatically selected: {selected_file.name}")
    
    # Load and visualize
    visualizer = SimpleSharpnessVisualizer()
    image_data, affine, header = visualizer.load_and_validate_nifti(selected_file)
    
    if image_data is not None:
        mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
        visualizer.quick_visualization(image_data, mask, title=f"Selected: {selected_file.name}")
        
        return image_data, affine, header, mask, selected_file
    
    return None


def example_3_full_sharpness_analysis():
    """
    Example 3: Complete sharpness analysis workflow
    """
    print("="*70)
    print("EXAMPLE 3: COMPLETE SHARPNESS ANALYSIS WORKFLOW")
    print("="*70)
    
    # Load a NIfTI file (you need to specify the actual path)
    nifti_file_path = "/path/to/your/brain.nii.gz"
    
    # For demo purposes, create synthetic data if real file not available
    if not Path(nifti_file_path).exists():
        print("Real file not found, creating synthetic brain data for demonstration...")
        image_data, affine = create_synthetic_brain_data()
        print("? Created synthetic brain data")
    else:
        visualizer = SimpleSharpnessVisualizer()
        image_data, affine, header = visualizer.load_and_validate_nifti(nifti_file_path)
        
        if image_data is None:
            print("Failed to load real data, falling back to synthetic data...")
            image_data, affine = create_synthetic_brain_data()
    
    # Now run the full sharpness analysis
    print("\nRunning complete 3D sharpness analysis...")
    
    try:
        # Try to import the full visualizer
        if PLOTLY_AVAILABLE:
            # If plotly is available, try full 3D visualization
            print("Plotly available - attempting full 3D visualization...")
            
            # For this demo, we'll use the simple visualizer since we don't have the full 3D class imported
            visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=6, num_patches=500)
            
            # Create threshold mask
            mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
            
            # Comprehensive analysis
            analysis_results = visualizer.comprehensive_analysis(
                image_data, mask, title="Complete Sharpness Analysis"
            )
            
            print("\n? Complete sharpness analysis finished!")
            print("Note: For full 3D interactive visualization, please ensure the")
            print("SharpnessVisualizer3D class from the previous code is available.")
            
            return image_data, analysis_results
        
        else:
            print("Plotly not available. Running basic analysis...")
            
            # Fallback to basic analysis
            visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=6, num_patches=500)
            mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
            analysis_results = visualizer.comprehensive_analysis(
                image_data, mask, title="Basic Analysis"
            )
            
            return image_data, analysis_results
        
    except Exception as e:
        print(f"Error in full analysis: {e}")
        print("Running basic visualization instead...")
        
        # Fallback to basic visualization
        visualizer = SimpleSharpnessVisualizer()
        mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
        analysis_results = visualizer.comprehensive_analysis(image_data, mask, title="Basic Analysis")
        
        return image_data, analysis_results


def example_4_batch_file_loading():
    """
    Example 4: Load multiple NIfTI files for batch analysis with mean sharpness
    """
    print("="*70)
    print("EXAMPLE 4: BATCH NIFTI FILE LOADING WITH SHARPNESS ANALYSIS")
    print("="*70)
    
    # SPECIFY YOUR FOLDER PATH HERE
    folder_path = "/path/to/your/nifti/folder"
    
    if not Path(folder_path).exists():
        print(f"Folder not found: {folder_path}")
        print("Please update the folder path to point to your NIfTI files.")
        
        # Demo with synthetic data
        print("\nCreating synthetic data for batch demo...")
        batch_results = []
        
        for i in range(3):
            print(f"\nCreating synthetic brain {i+1}/3...")
            image_data, affine = create_synthetic_brain_data()
            
            # Add different levels of blur to simulate different atlas qualities
            if i == 1:
                image_data = gaussian_filter(image_data, sigma=0.5)
            elif i == 2:
                image_data = gaussian_filter(image_data, sigma=1.0)
            
            # Initialize visualizer
            visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=8, num_patches=500)
            
            # Create mask and analyze
            mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
            analysis_results = visualizer.comprehensive_analysis(
                image_data, mask, title=f"Synthetic_Brain_{i+1}"
            )
            
            batch_results.append(analysis_results)
        
        # Print batch comparison
        print_batch_comparison(batch_results)
        return batch_results
    
    # Find all NIfTI files
    nifti_files = []
    for pattern in ['*.nii', '*.nii.gz']:
        nifti_files.extend(list(Path(folder_path).rglob(pattern)))
    
    nifti_files = sorted(list(set(nifti_files)))
    
    if len(nifti_files) == 0:
        print("No NIfTI files found!")
        return None
    
    print(f"Found {len(nifti_files)} NIfTI files for batch processing")
    
    # Initialize visualizer
    visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=8, num_patches=1000)
    
    # Process each file
    batch_results = []
    
    for i, file_path in enumerate(nifti_files[:5]):  # Process first 5 files
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{min(5, len(nifti_files))}: {file_path.name}")
        print(f"{'='*60}")
        
        # Load file
        image_data, affine, header = visualizer.load_and_validate_nifti(file_path)
        
        if image_data is not None:
            # Create mask and perform comprehensive analysis
            mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
            analysis_results = visualizer.comprehensive_analysis(
                image_data, mask, title=file_path.name
            )
            
            batch_results.append(analysis_results)
            print(f"  ? Processed successfully")
        else:
            print(f"  ? Failed to process")
    
    # Print batch comparison
    if batch_results:
        print_batch_comparison(batch_results)
    
    return batch_results


def print_batch_comparison(batch_results):
    """Print comparison table for batch results."""
    if not batch_results:
        return
    
    print(f"\n{'='*80}")
    print("BATCH SHARPNESS COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'File Name':<25} {'Shape':<15} {'Mask %':<8} {'Norm.Std':<10} {'Gradient':<10} {'Laplacian':<10}")
    print("-" * 80)
    
    for result in batch_results:
        title = result['title'][:24]  # Truncate long names
        shape_str = f"{result['basic_stats']['shape']}"[:14]
        mask_pct = result['mask_stats']['mask_coverage_percent']
        
        # Get sharpness values
        sharpness = result['sharpness_metrics']
        norm_std = sharpness['normalized_std']['mean']
        gradient = sharpness['gradient']['mean']
        laplacian = sharpness['laplacian']['mean']
        
        print(f"{title:<25} {shape_str:<15} {mask_pct:<8.1f} {norm_std:<10.4f} {gradient:<10.4f} {laplacian:<10.4f}")
    
    # Find best performers
    print(f"\n{'='*50}")
    print("BEST PERFORMERS")
    print(f"{'='*50}")
    
    # Best by normalized std sharpness
    best_norm_std = max(batch_results, key=lambda x: x['sharpness_metrics']['normalized_std']['mean'])
    print(f"Highest Normalized Std Sharpness: {best_norm_std['title']}")
    print(f"  Score: {best_norm_std['sharpness_metrics']['normalized_std']['mean']:.4f}")
    
    # Best by gradient sharpness
    best_gradient = max(batch_results, key=lambda x: x['sharpness_metrics']['gradient']['mean'])
    print(f"Highest Gradient Sharpness: {best_gradient['title']}")
    print(f"  Score: {best_gradient['sharpness_metrics']['gradient']['mean']:.4f}")
    
    # Calculate average scores
    avg_norm_std = np.mean([r['sharpness_metrics']['normalized_std']['mean'] for r in batch_results])
    avg_gradient = np.mean([r['sharpness_metrics']['gradient']['mean'] for r in batch_results])
    avg_laplacian = np.mean([r['sharpness_metrics']['laplacian']['mean'] for r in batch_results])
    
    print(f"\nAverage Scores:")
    print(f"  Normalized Std: {avg_norm_std:.4f}")
    print(f"  Gradient: {avg_gradient:.4f}")
    print(f"  Laplacian: {avg_laplacian:.4f}")


def create_synthetic_brain_data():
    """
    Create synthetic 3D brain data for demonstration purposes.
    """
    shape = (80, 80, 80)
    
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.linspace(-2, 2, shape[0]),
        np.linspace(-2, 2, shape[1]), 
        np.linspace(-2, 2, shape[2]),
        indexing='ij'
    )
    
    # Create brain-like structure
    brain_signal = (
        0.8 * np.exp(-(x**2 + y**2 + z**2) / 1.2) +  # Main brain volume
        0.3 * np.sin(3*np.pi*x) * np.cos(3*np.pi*y) * np.sin(3*np.pi*z) +  # Texture
        0.2 * np.exp(-((x-0.6)**2 + (y-0.6)**2 + z**2) / 0.4) +  # Right structure
        0.2 * np.exp(-((x+0.6)**2 + (y+0.6)**2 + z**2) / 0.4)   # Left structure
    )
    
    # Add noise
    noise = 0.03 * np.random.randn(*shape)
    synthetic_brain = brain_signal + noise
    
    # Normalize to [0, 1] range
    synthetic_brain = (synthetic_brain - np.min(synthetic_brain)) / (np.max(synthetic_brain) - np.min(synthetic_brain))
    
    # Create simple affine matrix
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0  # 2mm voxel size
    
    return synthetic_brain, affine


def interactive_file_selection():
    """
    Interactive file selection for loading NIfTI files.
    """
    print("="*70)
    print("INTERACTIVE NIFTI FILE SELECTION")
    print("="*70)
    
    # Get folder path from user
    folder_path = input("Enter the folder path containing NIfTI files: ").strip()
    
    if not folder_path:
        print("No path provided. Using current directory...")
        folder_path = "."
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return None
    
    # Find NIfTI files
    nifti_files = []
    for pattern in ['*.nii', '*.nii.gz']:
        nifti_files.extend(list(folder_path.glob(pattern)))
        nifti_files.extend(list(folder_path.rglob(pattern)))
    
    nifti_files = sorted(list(set(nifti_files)))
    
    if len(nifti_files) == 0:
        print("No NIfTI files found in the specified folder.")
        return None
    
    # Display available files
    print(f"\nFound {len(nifti_files)} NIfTI files:")
    for i, file_path in enumerate(nifti_files):
        file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  {i+1:2d}. {file_path.name} ({file_size:.1f} MB)")
    
    # Get user selection
    try:
        choice = input(f"\nSelect a file (1-{len(nifti_files)}) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("Exiting...")
            return None
        
        file_index = int(choice) - 1
        
        if 0 <= file_index < len(nifti_files):
            selected_file = nifti_files[file_index]
            print(f"\nSelected: {selected_file.name}")
            
            # Load and analyze the selected file
            visualizer = SimpleSharpnessVisualizer()
            image_data, affine, header = visualizer.load_and_validate_nifti(selected_file)
            
            if image_data is not None:
                # Get threshold from user
                threshold_input = input("\nEnter threshold value (0.0-1.0) or press Enter for default (0.15): ").strip()
                
                try:
                    threshold = float(threshold_input) if threshold_input else 0.15
                    threshold = max(0.0, min(1.0, threshold))  # Clamp to valid range
                except:
                    threshold = 0.15
                
                # Create mask and visualize
                mask = visualizer.create_threshold_mask(image_data, threshold=threshold)
                visualizer.quick_visualization(image_data, mask, title=f"Analysis: {selected_file.name}")
                
                return image_data, affine, header, mask, selected_file
            
        else:
            print("Invalid selection!")
            return None
            
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or interrupted by user.")
        return None


# Quick start functions with sharpness analysis
def quick_load_and_visualize(file_path, threshold=0.15, compute_sharpness=True):
    """
    Quick function to load and visualize a NIfTI file with sharpness analysis.
    
    Parameters:
    -----------
    file_path : str
        Path to the NIfTI file
    threshold : float
        Threshold for mask creation (default: 0.15)
    compute_sharpness : bool
        Whether to compute sharpness metrics (default: True)
    
    Returns:
    --------
    tuple
        (image_data, affine, header, mask, sharpness_results) or None if failed
    """
    visualizer = SimpleSharpnessVisualizer(patch_size=7, grid_spacing=8, num_patches=3000)
    
    # Load file
    image_data, affine, header = visualizer.load_and_validate_nifti(file_path)
    
    if image_data is not None:
        # Create mask
        mask = visualizer.create_threshold_mask(image_data, threshold=threshold)
        
        if compute_sharpness:
            # Comprehensive analysis with sharpness
            analysis_results = visualizer.comprehensive_analysis(
                image_data, mask, title=f"Quick Load: {Path(file_path).name}"
            )
            
            return image_data, affine, header, mask, analysis_results
        else:
            # Just visualization
            visualizer.quick_visualization(image_data, mask, title=f"Quick Load: {Path(file_path).name}")
            return image_data, affine, header, mask, None
    
    return None


def quick_folder_analysis(folder_path, max_files=5, compute_sharpness=True):
    """
    Quick function to analyze multiple files in a folder with sharpness computation.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing NIfTI files
    max_files : int
        Maximum number of files to process
    compute_sharpness : bool
        Whether to compute sharpness metrics
    
    Returns:
    --------
    list
        List of analysis results with sharpness metrics
    """
    folder_path = Path(folder_path)
    
    # Find files
    nifti_files = []
    for pattern in ['*.nii', '*.nii.gz']:
        nifti_files.extend(list(folder_path.glob(pattern)))
    
    nifti_files = sorted(list(set(nifti_files)))[:max_files]
    
    if len(nifti_files) == 0:
        print(f"No NIfTI files found in {folder_path}")
        return []
    
    print(f"Processing {len(nifti_files)} files...")
    
    results = []
    visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=8, num_patches=500)
    
    for file_path in nifti_files:
        print(f"\nProcessing: {file_path.name}")
        
        image_data, affine, header = visualizer.load_and_validate_nifti(file_path)
        
        if image_data is not None:
            mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
            
            if compute_sharpness:
                # Comprehensive analysis
                analysis_results = visualizer.comprehensive_analysis(
                    image_data, mask, title=file_path.name
                )
                results.append(analysis_results)
            else:
                # Basic analysis
                result = {
                    'file_name': file_path.name,
                    'shape': image_data.shape,
                    'data_range': [float(np.min(image_data)), float(np.max(image_data))],
                    'mask_coverage': (np.sum(mask) / np.prod(image_data.shape)) * 100
                }
                results.append(result)
    
    # Print comparison if sharpness was computed
    if compute_sharpness and results:
        print_batch_comparison(results)
    
    return results


def compare_sharpness_methods(file_path, thresholds=[0.1, 0.15, 0.2, 0.25]):
    """
    Compare sharpness results using different threshold values.
    
    Parameters:
    -----------
    file_path : str
        Path to NIfTI file
    thresholds : list
        List of threshold values to test
        
    Returns:
    --------
    dict
        Comparison results for different thresholds
    """
    print(f"Comparing sharpness methods for: {Path(file_path).name}")
    print("="*60)
    
    visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=8, num_patches=1000)
    
    # Load file once
    image_data, affine, header = visualizer.load_and_validate_nifti(file_path)
    
    if image_data is None:
        print("Failed to load file!")
        return None
    
    comparison_results = {}
    
    # Test each threshold
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        
        # Create mask with this threshold
        mask = visualizer.create_threshold_mask(image_data, threshold=threshold)
        
        # Compute sharpness metrics
        sharpness_results = visualizer.compute_mean_sharpness(image_data, mask, method='all')
        
        # Store results
        comparison_results[f'threshold_{threshold}'] = {
            'threshold': threshold,
            'mask_coverage': (np.sum(mask) / np.prod(image_data.shape)) * 100,
            'sharpness_metrics': sharpness_results
        }
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("THRESHOLD COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Threshold':<10} {'Coverage %':<12} {'Norm.Std':<10} {'Gradient':<10} {'Laplacian':<10}")
    print("-" * 80)
    
    for key, result in comparison_results.items():
        thresh = result['threshold']
        coverage = result['mask_coverage']
        norm_std = result['sharpness_metrics']['normalized_std']['mean']
        gradient = result['sharpness_metrics']['gradient']['mean']
        laplacian = result['sharpness_metrics']['laplacian']['mean']
        
        print(f"{thresh:<10.2f} {coverage:<12.1f} {norm_std:<10.4f} {gradient:<10.4f} {laplacian:<10.4f}")
    
    # Find optimal threshold (highest normalized std sharpness with reasonable coverage)
    best_threshold = None
    best_score = 0
    
    for key, result in comparison_results.items():
        coverage = result['mask_coverage']
        norm_std = result['sharpness_metrics']['normalized_std']['mean']
        
        # Only consider thresholds with at least 10% coverage
        if coverage >= 10.0 and norm_std > best_score:
            best_score = norm_std
            best_threshold = result['threshold']
    
    if best_threshold is not None:
        print(f"\nRecommended threshold: {best_threshold}")
        print(f"Best normalized std sharpness: {best_score:.4f}")
    
    return comparison_results


def save_sharpness_results_to_csv(results, output_path="sharpness_analysis_results.csv"):
    """
    Save sharpness analysis results to CSV file.
    
    Parameters:
    -----------
    results : list or dict
        Analysis results from batch processing or single file
    output_path : str
        Path to save CSV file
    """
    print(f"Saving results to: {output_path}")
    
    # Handle different input formats
    if isinstance(results, dict):
        # Single file results
        results = [results]
    
    # Extract data for CSV
    csv_data = []
    
    for result in results:
        if 'sharpness_metrics' in result:
            # Full analysis results
            basic = result['basic_stats']
            mask = result['mask_stats']
            sharpness = result['sharpness_metrics']
            
            row = {
                'file_name': result['title'],
                'shape_x': basic['shape'][0],
                'shape_y': basic['shape'][1],
                'shape_z': basic['shape'][2],
                'total_voxels': basic['total_voxels'],
                'value_min': basic['value_range'][0],
                'value_max': basic['value_range'][1],
                'mean_intensity': basic['mean_intensity'],
                'std_intensity': basic['std_intensity'],
                'mask_coverage_percent': mask['mask_coverage_percent'],
                'masked_voxels': mask['masked_voxels'],
                'norm_std_mean': sharpness['normalized_std']['mean'],
                'norm_std_std': sharpness['normalized_std']['std'],
                'norm_std_min': sharpness['normalized_std']['min'],
                'norm_std_max': sharpness['normalized_std']['max'],
                'norm_std_median': sharpness['normalized_std']['median'],
                'norm_std_samples': sharpness['normalized_std']['samples'],
                'gradient_mean': sharpness['gradient']['mean'],
                'gradient_std': sharpness['gradient']['std'],
                'gradient_min': sharpness['gradient']['min'],
                'gradient_max': sharpness['gradient']['max'],
                'laplacian_mean': sharpness['laplacian']['mean'],
                'laplacian_std': sharpness['laplacian']['std'],
                'laplacian_min': sharpness['laplacian']['min'],
                'laplacian_max': sharpness['laplacian']['max']
            }
            
            csv_data.append(row)
    
    # Create DataFrame and save
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        print(f"? Saved {len(csv_data)} results to {output_path}")
        print(f"Columns: {list(df.columns)}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Mean Normalized Std Sharpness: {df['norm_std_mean'].mean():.4f} � {df['norm_std_mean'].std():.4f}")
        print(f"Mean Gradient Sharpness: {df['gradient_mean'].mean():.4f} � {df['gradient_mean'].std():.4f}")
        print(f"Mean Laplacian Sharpness: {df['laplacian_mean'].mean():.4f} � {df['laplacian_mean'].std():.4f}")
        
        return df
    else:
        print("No valid data to save!")
        return None


def create_sharpness_comparison_plot(results, save_path="sharpness_comparison.png"):
    """
    Create comparison plots for sharpness results.
    
    Parameters:
    -----------
    results : list
        List of analysis results
    save_path : str
        Path to save the plot
    """
    if not results:
        print("No results to plot!")
        return
    
    # Extract data for plotting
    file_names = []
    norm_std_values = []
    gradient_values = []
    laplacian_values = []
    coverage_values = []
    
    for result in results:
        if 'sharpness_metrics' in result:
            file_names.append(result['title'][:15] + '...' if len(result['title']) > 15 else result['title'])
            norm_std_values.append(result['sharpness_metrics']['normalized_std']['mean'])
            gradient_values.append(result['sharpness_metrics']['gradient']['mean'])
            laplacian_values.append(result['sharpness_metrics']['laplacian']['mean'])
            coverage_values.append(result['mask_stats']['mask_coverage_percent'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Normalized Std Sharpness
    axes[0, 0].bar(range(len(file_names)), norm_std_values, alpha=0.7, color='blue')
    axes[0, 0].set_title('Normalized Std Sharpness')
    axes[0, 0].set_ylabel('Sharpness Score')
    axes[0, 0].set_xticks(range(len(file_names)))
    axes[0, 0].set_xticklabels(file_names, rotation=45, ha='right')
    
    # Gradient Sharpness
    axes[0, 1].bar(range(len(file_names)), gradient_values, alpha=0.7, color='red')
    axes[0, 1].set_title('Gradient Magnitude Sharpness')
    axes[0, 1].set_ylabel('Sharpness Score')
    axes[0, 1].set_xticks(range(len(file_names)))
    axes[0, 1].set_xticklabels(file_names, rotation=45, ha='right')
    
    # Laplacian Sharpness
    axes[1, 0].bar(range(len(file_names)), laplacian_values, alpha=0.7, color='green')
    axes[1, 0].set_title('Laplacian Sharpness')
    axes[1, 0].set_ylabel('Sharpness Score')
    axes[1, 0].set_xticks(range(len(file_names)))
    axes[1, 0].set_xticklabels(file_names, rotation=45, ha='right')
    
    # Coverage vs Normalized Std Sharpness
    axes[1, 1].scatter(coverage_values, norm_std_values, alpha=0.7, s=60)
    axes[1, 1].set_xlabel('Mask Coverage (%)')
    axes[1, 1].set_ylabel('Normalized Std Sharpness')
    axes[1, 1].set_title('Coverage vs Sharpness')
    
    # Add file labels to scatter plot
    for i, name in enumerate(file_names):
        axes[1, 1].annotate(name, (coverage_values[i], norm_std_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to: {save_path}")


# Enhanced main function with sharpness analysis
def main():
    """
    Main function demonstrating different ways to load NIfTI files with sharpness analysis.
    """
    print("NIfTI FILE LOADING WITH SHARPNESS ANALYSIS")
    print("="*80)
    
    print("\nChoose an analysis option:")
    print("1. Load single NIfTI file with sharpness analysis")
    print("2. Browse folder for NIfTI files with sharpness analysis")
    print("3. Complete sharpness analysis workflow")
    print("4. Batch file loading with sharpness comparison")
    print("5. Interactive file selection with sharpness analysis")
    print("6. Create synthetic data demo with sharpness analysis")
    print("7. Compare different threshold values")
    print("8. Quick sharpness analysis (specify file path)")
    
    try:
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            print("RUNNING EXAMPLE 1: SINGLE FILE WITH SHARPNESS")
            print("="*50)
            print("IMPORTANT: Update the file path in example_1_single_file_loading() function")
            result = example_1_single_file_loading()
            
            if result is not None:
                # Save results to CSV
                image_data, affine, header, mask, analysis_results = result
                save_sharpness_results_to_csv([analysis_results], "single_file_sharpness.csv")
            
        elif choice == "2":
            print("\n" + "="*50)
            print("RUNNING EXAMPLE 2: FOLDER BROWSING WITH SHARPNESS")
            print("="*50)
            print("IMPORTANT: Update the folder path in example_2_folder_browsing() function")
            result = example_2_folder_browsing()
            
        elif choice == "3":
            print("\n" + "="*50)
            print("RUNNING EXAMPLE 3: COMPLETE ANALYSIS")
            print("="*50)
            result = example_3_full_sharpness_analysis()
            
        elif choice == "4":
            print("\n" + "="*50)
            print("RUNNING EXAMPLE 4: BATCH LOADING WITH SHARPNESS")
            print("="*50)
            result = example_4_batch_file_loading()
            
            if result:
                # Save batch results
                save_sharpness_results_to_csv(result, "batch_sharpness_results.csv")
                create_sharpness_comparison_plot(result, "batch_sharpness_comparison.png")
            
        elif choice == "5":
            print("\n" + "="*50)
            print("RUNNING EXAMPLE 5: INTERACTIVE SELECTION WITH SHARPNESS")
            print("="*50)
            result = interactive_file_selection()
            
        elif choice == "6":
            print("\n" + "="*50)
            print("RUNNING EXAMPLE 6: SYNTHETIC DATA WITH SHARPNESS")
            print("="*50)
            image_data, affine = create_synthetic_brain_data()
            visualizer = SimpleSharpnessVisualizer(patch_size=5, grid_spacing=6, num_patches=500)
            mask = visualizer.create_threshold_mask(image_data, threshold=0.15)
            analysis_results = visualizer.comprehensive_analysis(image_data, mask, title="Synthetic Brain Data")
            
            # Save synthetic results
            save_sharpness_results_to_csv([analysis_results], "synthetic_sharpness.csv")
            result = (image_data, affine, mask, analysis_results)
            
        elif choice == "7":
            file_path = input("Enter path to NIfTI file for threshold comparison: ").strip()
            if Path(file_path).exists():
                result = compare_sharpness_methods(file_path)
            else:
                print("File not found! Using synthetic data for demo...")
                # Create synthetic data and save as temporary file
                image_data, affine = create_synthetic_brain_data()
                temp_path = "temp_synthetic.nii.gz"
                nii_img = nib.Nifti1Image(image_data, affine)
                nib.save(nii_img, temp_path)
                result = compare_sharpness_methods(temp_path)
                # Clean up
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
            
        elif choice == "8":
            file_path = input("Enter path to NIfTI file: ").strip()
            if Path(file_path).exists():
                result = quick_load_and_visualize(file_path, threshold=0.10, compute_sharpness=True)
                if result is not None:
                    image_data, affine, header, mask, analysis_results = result
                    save_sharpness_results_to_csv([analysis_results], "quick_analysis_sharpness.csv")
            else:
                print("File not found!")
                result = None
            
        else:
            print("Invalid choice. Running interactive selection...")
            result = interactive_file_selection()
        
        if result is not None:
            print("\n? Analysis completed successfully!")
            print("Check the generated CSV files and plots for detailed results.")
        else:
            print("\n? Analysis failed or was cancelled.")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies first
    normalized_std = []
    
    for i in range(4,19):
        
        #atlas_path = '/media/ch237087/LaCie/Geodesic_Distance/atlas_male_hip_raw/'+'check_atlas_target_'+str(i)+'_rest_LOOCLSIG/' + 'atlas_950_1.nii.gz'
        atlas_path = './check_atlas_male/check_atlas_target_'+str(i)+'_rest_LOOCLSIG' + '/atlas_950_1.nii.gz' 
        result1 = quick_load_and_visualize(atlas_path)
        normalized_std.append(result1[4]['sharpness_metrics']['normalized_std']['mean'])
    print("normalized_std for all the cases:", np.mean(normalized_std))
    scipy.io.savemat("sharpness_atlas_CL.mat", {"normalized_std": normalized_std})
    #result1 = quick_load_and_visualize("/media/ch237087/LaCie/Contrastive Deep Learning/check_atlas_male/Download_from_Turing/check_atlas_target_13_rest_4_to_12_CL_0/atlas_950_1.nii.gz")
    
    #result2 = quick_load_and_visualize("/media/ch237087/LaCie/Contrastive Deep Learning/check_atlas_male/Download_from_Turing/check_atlas_target_13_rest_4_to_12_CL_1/atlas_950_1.nii.gz")
    
    #result3 = quick_load_and_visualize("/media/ch237087/LaCie/Contrastive Deep Learning/check_atlas_male/Download_from_Turing/check_atlas_target_13_rest_4_to_12_CL_2/atlas_950_1.nii.gz")
    
    #result4 = quick_load_and_visualize("/media/ch237087/LaCie/Contrastive Deep Learning/check_atlas_male/Download_from_Turing/check_atlas_target_13_rest_4_to_12_CL_3/atlas_950_1.nii.gz")
    
    #result5 = quick_load_and_visualize("/media/ch237087/LaCie/Contrastive Deep Learning/check_atlas_male/Download_from_Turing/check_atlas_target_13_rest_4_to_12_CL_4/atlas_950_1.nii.gz")
    
    #result6 = quick_load_and_visualize("/media/ch237087/LaCie/Contrastive Deep Learning/check_atlas_male/check_atlas_target_13_rest_4_to_12_CL_5/atlas_950_3.nii.gz")