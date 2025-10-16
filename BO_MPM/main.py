import argparse
from scripts.run_optimization import run_optimization
from scripts.visualize_results import visualize_results

def main():
    parser = argparse.ArgumentParser(description='MPM Parameter Optimization Tool')
    subparsers = parser.add_subparsers(dest='command')
    
    # Optimization command
    optimize_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
<<<<<<< Updated upstream:BO_MPM/main.py
    optimize_parser.add_argument('--width', type=float, default=3.9, help='Container width')
    optimize_parser.add_argument('--height', type=float, default=6.8, help='Container height')
    optimize_parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    optimize_parser.add_argument('--seed', type=int, default=42, help='Random seed')
=======
    # optimize_parser.add_argument('--width', type=float, default=3.9, help='Container width')
    # optimize_parser.add_argument('--height', type=float, default=6.8, help='Container height')
    optimize_parser.add_argument('--evals', type=int, default=10, help='Total number of evaluations.')
    optimize_parser.add_argument('--init_points', type=int, default=5, help='Number of initial LHS points.')
    optimize_parser.add_argument('--batch_size', type=int, default=5, help='Number of parallel evaluations per batch (q).')
    optimize_parser.add_argument('--seed', type=int, default=66, help='Random seed')
>>>>>>> Stashed changes:main.py
    optimize_parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    # Visualization command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize results')
    visualize_parser.add_argument('file', type=str, help='Path to result file')
    
    args = parser.parse_args()
    
    if args.command == 'optimize':
        run_optimization(
            width=args.width,
            height=args.height,
            sampling_number=args.samples,
            seed=args.seed,
            output_dir=args.output
        )
    elif args.command == 'visualize':
        visualize_results(args.file)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()