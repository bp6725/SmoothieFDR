"""
Main script to run spatial FDR evaluation on ADbench datasets.

Usage:
    python main.py --datasets 25_musk --n_samples 500 --n_reps 100
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import warnings
import os
import sys


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spatial_fdr_evaluation.experiments.run_evaluation import (
    run_evaluation,
    run_lambda_sensitivity
)
from third_party.ADbench.data_generator import DataGenerator

class AdBench_Simulator():
    def __init__(self):
      pass

    @staticmethod
    def load_datasets(data_sets=None, params = {}):
        '''
        We return list of datasets. For each dataset we return 1.'X_train' 2.'X_test' 3. 'Y_test' 4. 'name' 5.'params'
        :param data_sets: datasets names in list
        :param params:
        :return:
        '''
        if data_sets is None:
            data_sets = ['2_annthyroid']

        all_sets = []
        for data_set_name in data_sets :
            set = AdBench_Simulator.load_dataset(data_set_name, params['la'])

            all_sets.append(set)

        return all_sets

    @staticmethod
    def load_dataset(data_set = '2_annthyroid', la = 0.1,resample = True, return_all_data = False):
        data_generator = DataGenerator(dataset = data_set,generate_duplicates=resample)
        data = data_generator.generator(la =la,return_all_data=return_all_data)
        return data


def load_from_ADbench(dataset_name, n_dims_to_take = -1, n_samples = -1, la = 0.1) :
    dataset = AdBench_Simulator.load_dataset(dataset_name, la,True,False)
    X_train, X_test = dataset['X_train'], dataset['X_test']
    Y_train, Y_test = dataset['y_train'], dataset['y_test']

    if n_samples != -1:
        X_train = X_train[0:n_samples]

    if n_dims_to_take != -1:
        raise NotImplementedError('No other dims for you.')

    #normliaze X_train and X_test
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    running_results = {}
    running_results['X_train'] = X_train
    running_results['X_test'] = X_test
    running_results['Y_train'] = Y_train
    running_results['Y_test'] = Y_test
    running_results['dataset_name'] = dataset_name
    running_results["is supervised"] = sum(Y_train) > 0
    running_results["dim"] = X_train.shape[1]

    return running_results


def main():
    parser = argparse.ArgumentParser(
        description='Run spatial FDR evaluation on ADbench datasets'
    )
    
    # parser.add_argument(
    #     '--datasets',
    #     nargs='+',
    #     default=['7_Cardiotocography', '24_mnist', '11_donors', '36_speech', '46_WPBC', '18_Ionosphere', '13_fraud',
    #         '20_letter',
    #         '32_shuttle', '22_magic.gamma',
    #         '35_SpamBase', '43_WDBC', '10_cover', '19_landsat', '14_glass', '23_mammography', '26_optdigits',
    #         '31_satimage-2', '30_satellite', '42_WBC',
    #         '3_backdoor', '27_PageBlocks', '47_yeast', '21_Lymphography', '41_Waveform', '44_Wilt', '2_annthyroid',
    #         '37_Stamps', '38_thyroid', '39_vertebral',
    #         '8_celeba', '28_pendigits', '9_census', '25_musk', '34_smtp', '29_Pima',
    #         '15_Hepatitis', '45_wine', '33_skin', '6_cardio', '1_ALOI', '17_InternetAds', '40_vowels', '4_breastw',
    #         '16_http', '5_campaign', '12_fault'],
    #     help='List of dataset names to evaluate'
    # )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['7_Cardiotocography', '24_mnist', '11_donors'],
        help='List of dataset names to evaluate'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=500,
        help='Number of spatial locations to sample per replication'
    )
    
    parser.add_argument(
        '--n_reps',
        type=int,
        default=3,
        help='Number of Monte Carlo replications'
    )
    
    parser.add_argument(
        '--spatial_strengths',
        nargs='+',
        default=['none', 'weak', 'medium', 'strong'],
        help='Spatial clustering strengths to evaluate'
    )
    
    parser.add_argument(
        '--effect_strength',
        type=str,
        default='medium',
        choices=['weak', 'medium', 'strong'],
        help='Signal effect strength'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--lambda_sensitivity',
        action='store_true',
        help='Run lambda sensitivity analysis'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Run evaluation on each dataset
    for dataset_name in args.datasets:
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)
        
        # try:
        if True :
            # Load data
            print(f"Loading dataset: {dataset_name}")
            data_dict = load_from_ADbench(dataset_name)
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            
            print(f"  Train shape: {X_train.shape}")
            print(f"  Test shape: {X_test.shape}")
            
            # Use training data for spatial structure
            X_data = np.concatenate([X_train,X_test])
            
            # Create dataset-specific output directory
            output_dir = Path(args.output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Main evaluation
            print(f"\nRunning main evaluation...")
            results = run_evaluation(
                X_data=X_data,
                n_samples=args.n_samples,
                spatial_strengths=args.spatial_strengths,
                effect_strength=args.effect_strength,
                n_replications=args.n_reps,
                output_dir=str(output_dir),
                random_state=args.random_state
            )
            
            # Lambda sensitivity analysis (optional)
            if args.lambda_sensitivity:
                print(f"\nRunning lambda sensitivity analysis...")
                sensitivity_results = run_lambda_sensitivity(
                    X_data=X_data,
                    n_samples=args.n_samples,
                    spatial_strength='medium',
                    effect_strength=args.effect_strength,
                    n_replications=20,
                    output_dir=str(output_dir),
                    random_state=args.random_state
                )
            
            print(f"\n✓ Completed evaluation for {dataset_name}")
            print(f"  Results saved to: {output_dir}")
            
        # except Exception as e:
        #     print(f"\n✗ Error processing {dataset_name}: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     continue
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
