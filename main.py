import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Import the new sampler and refactored evaluator
from spatial_fdr_evaluation.data.structural_sampling import (
    load_and_sample_structural_data,
    Valid_Datasets
)
from spatial_fdr_evaluation.experiments.run_evaluation import evaluate_fixed_sample
from spatial_fdr_evaluation.visualization.plots import (
    visualize_structural_groups,
    plot_dataset_overview,
    plot_kernel_diagnostics,
    plot_method_comparison_detail
)


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial FDR Evaluation on Structural Data")
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--n_seeds', type=int, default=5, help='Number of random seeds per dataset')
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.results_dir, f"structural_experiment_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # 1. Define Sampling Difficulty (The 4 Groups)
    sampling_config = {
        'n_h1': 50,
        'n_h0_compact': 150,
        'n_h0_close': 20,
        'n_h0_far': 280,
        'sigma_factor': 0.5,
        'effect_strength': 'medium'
    }

    # 2. Define Methods to Test
    base_methods_config = {
        'BH': {'alpha': 0.1},
        'SpatialFDR': {
            'kernel_type': 'matern',
            'lambda_reg': 5.0,
            'lambda_bound': 100.0,
            'kernel_params': {'nu': 1.5},
            'optimizer': 'natural_gradient',
            'alpha': 0.1
        }
    }

    print(f"Starting Evaluation on {len(Valid_Datasets)} Datasets...")

    all_results = []

    for dataset_name in Valid_Datasets:
        print(f"\n{'=' * 50}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 50}")

        # Create plot directory for this dataset
        plot_dir = os.path.join(result_dir, "plots", dataset_name)
        os.makedirs(plot_dir, exist_ok=True)

        dataset_results = []

        for i in range(args.n_seeds):
            seed = 42 + i
            is_viz_run = (i == 0)  # Only visualize the first seed

            print(f"  > Run {i + 1}/{args.n_seeds} (Seed {seed})...")

            # A. Generate Data
            np.random.seed(seed)
            data_tuple = load_and_sample_structural_data(
                dataset_name,
                **sampling_config
            )

            if data_tuple is None:
                print(f"    ! Failed to sample {dataset_name}. Skipping.")
                continue

            X, p_values, true_labels = data_tuple

            # --- VISUALIZATION BLOCK 1: DATA ---
            if is_viz_run:
                # 1. Dataset Overview
                plot_dataset_overview(
                    X, true_labels, p_values,
                    save_path=os.path.join(plot_dir, f"data_overview_seed{seed}.png")
                )

            # B. Evaluate
            # --- BUG FIX START ---
            # Handle return values correctly depending on is_viz_run
            if is_viz_run:
                metrics_dict, details_dict = evaluate_fixed_sample(
                    X, p_values, true_labels,
                    base_methods_config,
                    return_details=True
                )
            else:
                metrics_dict = evaluate_fixed_sample(
                    X, p_values, true_labels,
                    base_methods_config,
                    return_details=False
                )
                details_dict = {}
            # --- BUG FIX END ---

            # --- VISUALIZATION BLOCK 2: RESULTS ---
            if is_viz_run and 'SpatialFDR' in details_dict and 'BH' in details_dict:
                # 3. Method Comparison
                plot_method_comparison_detail(
                    X, true_labels, p_values,
                    lfdr=details_dict['SpatialFDR']['lfdr'],
                    rejections_spatial=details_dict['SpatialFDR']['rejections'],
                    rejections_bh=details_dict['BH']['rejections'],
                    stats_spatial=metrics_dict['SpatialFDR'],
                    stats_bh=metrics_dict['BH'],
                    save_path=os.path.join(plot_dir, f"comparison_seed{seed}.png")
                )

            # C. Format Results
            for method, metrics in metrics_dict.items():
                if metrics:
                    row = metrics.copy()
                    row['method'] = method
                    row['dataset'] = dataset_name
                    row['seed'] = seed
                    dataset_results.append(row)

        # Aggregate & Save Intermediate
        if dataset_results:
            df_dataset = pd.DataFrame(dataset_results)
            all_results.append(df_dataset)

            save_path = os.path.join(result_dir, f"{dataset_name}_results.csv")
            df_dataset.to_csv(save_path, index=False)

            summary = df_dataset.groupby('method')[['power', 'FDR']].mean()
            print("\n    Results Summary:")
            print(summary)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_path = os.path.join(result_dir, "FINAL_structural_results.csv")
        final_df.to_csv(final_path, index=False)
        print(f"\nDone! Full results saved to {final_path}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()