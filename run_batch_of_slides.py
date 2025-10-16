"""
Example usage:

```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""
import argparse
import csv
import multiprocessing as mp
import os
import shutil
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from trident import Processor
from trident.Concurrency import cache_batch
from trident.IO import collect_valid_slides
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry


def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the Trident processing script.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all Trident processing options.
    """
    parser = argparse.ArgumentParser(description='Run Trident')

    # Generic arguments 
    parser.add_argument('--gpu', '--gpus', dest='gpus', type=int, nargs='+', default=[0],
                        help='GPU indices to use for processing tasks. Provide one or multiple space-separated indices (e.g. --gpus 0 1).')
    parser.add_argument('--task', type=str, default='seg', 
                        choices=['seg', 'coords', 'feat', 'all'], 
                        help='Task to run: seg (segmentation), coords (save tissue coordinates), img (save tissue images), feat (extract features).')
    parser.add_argument('--job_dir', type=str, required=True, help='Directory to store outputs.')
    parser.add_argument('--skip_errors', action='store_true', default=False, 
                        help='Skip errored slides and continue processing.')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of workers. Set to 0 to use main process.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="Batch size used for segmentation and feature extraction. Will be override by"
                        "`seg_batch_size` and `feat_batch_size` if you want to use different ones. Defaults to 64.")

    # Caching argument for fast WSI processing
    parser.add_argument(
        '--wsi_cache', type=str, default=None,
        help='Path to a local cache (e.g., SSD) used to speed up access to WSIs stored on slower drives (e.g., HDD).'
    )
    parser.add_argument(
        '--cache_batch_size', type=int, default=32,
        help='Maximum number of slides to cache locally at once. Helps control disk usage.'
    )

    # Slide-related arguments
    parser.add_argument('--wsi_dir', type=str, required=True, 
                        help='Directory containing WSI files (no nesting allowed).')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None, 
                        help='List of allowed file extensions for WSI files.')
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--custom_list_of_wsis', type=str, default=None,
                    help='Custom list of WSIs specified in a csv file.')
    parser.add_argument('--reader_type', type=str, choices=['openslide', 'image', 'cucim', 'sdpc'], default=None,
                    help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim", "sdpc"]. Defaults to None (auto-determine which reader to use).')
    parser.add_argument("--search_nested", action="store_true",
                        help=("If set, recursively search for whole-slide images (WSIs) within all subdirectories of "
                              "`wsi_source`. Uses `os.walk` to include slides from nested folders. "
                              "This allows processing of datasets organized in hierarchical structures. "
                              "Defaults to False (only top-level slides are included)."))
    # Segmentation arguments 
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc'], 
                        help='Type of tissue vs background segmenter. Options are HEST or GrandQC.')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5, 
                    help='Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.')
    parser.add_argument('--remove_holes', action='store_true', default=False, 
                        help='Do you want to remove holes?')
    parser.add_argument('--remove_artifacts', action='store_true', default=False, 
                        help='Do you want to run an additional model to remove artifacts (including penmarks, blurs, stains, etc.)?')
    parser.add_argument('--remove_penmarks', action='store_true', default=False, 
                        help='Do you want to run an additional model to remove penmarks?')
    parser.add_argument('--seg_batch_size', type=int, default=None, 
                        help='Batch size for segmentation. Defaults to None (use `batch_size` argument instead).')
    
    # Patching arguments
    parser.add_argument('--mag', type=int, choices=[5, 10, 20, 40, 80], default=20, 
                        help='Magnification for coords/features extraction.')
    parser.add_argument('--patch_size', type=int, default=512, 
                        help='Patch size for coords/image extraction.')
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Absolute overlap for patching in pixels. Defaults to 0.')
    parser.add_argument('--min_tissue_proportion', type=float, default=0., 
                        help='Minimum proportion of the patch under tissue to be kept. Between 0. and 1.0. Defaults to 0.')
    parser.add_argument('--coords_dir', type=str, default=None, 
                        help='Directory to save/restore tissue coordinates.')
    
    # Feature extraction arguments 
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', 
                        choices=patch_encoder_registry.keys(),
                        help='Patch encoder to use')
    parser.add_argument(
        '--patch_encoder_ckpt_path', type=str, default=None,
        help=(
            "Optional local path to a patch encoder checkpoint (.pt, .pth, .bin, or .safetensors). "
            "This is only needed in offline environments (e.g., compute clusters without internet). "
            "If not provided, models are downloaded automatically from Hugging Face. "
            "You can also specify local paths via the model registry at "
            "`./trident/patch_encoder_models/local_ckpts.json`."
        )
    )
    parser.add_argument('--slide_encoder', type=str, default=None, 
                        choices=slide_encoder_registry.keys(), 
                        help='Slide encoder to use')
    parser.add_argument('--feat_batch_size', type=int, default=None, 
                        help='Batch size for feature extraction. Defaults to None (use `batch_size` argument instead).')
    return parser


def clone_args(base_args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    """Return a shallow copy of the provided namespace with updates applied."""
    data = vars(base_args).copy()
    data.update(overrides)
    return argparse.Namespace(**data)


def resolve_devices(args: argparse.Namespace) -> List[str]:
    """Derive the list of target devices from CLI arguments."""
    if torch.cuda.is_available():
        return [f'cuda:{idx}' for idx in args.gpus]
    if len(args.gpus) > 1:
        print('[MAIN] CUDA not available; using CPU despite multiple GPU indices being provided.')
    return ['cpu']


def slide_outputs_complete(slide_path: str, args: argparse.Namespace, task_sequence: Sequence[str]) -> bool:
    """Return True if all required outputs exist for the slide for the requested tasks."""
    slide_stem = os.path.splitext(os.path.basename(slide_path))[0]
    coords_dir = args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap'

    for task_name in task_sequence:
        if task_name == 'seg':
            if not os.path.exists(os.path.join(args.job_dir, 'contours', f'{slide_stem}.jpg')):
                return False
        elif task_name == 'coords':
            if not os.path.exists(os.path.join(args.job_dir, coords_dir, 'patches', f'{slide_stem}_patches.h5')):
                return False
        elif task_name == 'feat':
            # Check if feature file exists
            if args.slide_encoder is None:
                features_dir = os.path.join(args.job_dir, coords_dir, f'features_{args.patch_encoder}')
            else:
                features_dir = os.path.join(args.job_dir, coords_dir, f'slide_features_{args.slide_encoder}')
            
            if not features_dir or not os.path.isdir(features_dir):
                return False
            if not any(os.path.exists(os.path.join(features_dir, f'{slide_stem}.{ext}')) for ext in ('h5', 'pt')):
                return False
        else:
            return False
    return True


def filter_completed_slides(slide_paths: List[str], args: argparse.Namespace, task_sequence: Sequence[str]) -> List[str]:
    """Filter out slides whose outputs already exist for all requested tasks."""
    return [slide for slide in slide_paths if not slide_outputs_complete(slide, args, task_sequence)]


def cleanup_files(job_dir: str, cache_dir: Optional[str] = None) -> Tuple[int, int]:
    """
    Remove stale lock files and optionally clean cache directory.
    
    Returns
    -------
    Tuple[int, int]
        Number of lock files removed and cache items removed.
    """
    # Remove lock files
    lock_count = 0
    if os.path.isdir(job_dir):
        for root, _, files in os.walk(job_dir):
            for filename in files:
                if filename.endswith('.lock'):
                    try:
                        os.remove(os.path.join(root, filename))
                        lock_count += 1
                    except OSError:
                        pass
    
    # Clean cache directory
    cache_count = 0
    if cache_dir and os.path.isdir(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                cache_count += 1
            except OSError:
                pass
    
    return lock_count, cache_count


def load_custom_slide_rows(csv_path: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Load rows from a user-provided slide CSV so batches keep metadata like MPP."""

    with open(csv_path, newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'wsi' not in reader.fieldnames:
            raise ValueError(f"Custom slide list '{csv_path}' must include a 'wsi' column.")
        fieldnames = reader.fieldnames
        rows = {row['wsi']: dict(row) for row in reader}

    return rows, fieldnames


def write_batch_csv(
    rel_paths: Sequence[str],
    batch_id: int,
    root_dir: str,
    rows_by_wsi: Optional[Dict[str, Dict[str, str]]],
    fieldnames: Sequence[str],
    source_csv: Optional[str] = None,
) -> str:
    """Persist a per-batch CSV listing slides for Processor consumption."""
    os.makedirs(root_dir, exist_ok=True)
    csv_path = os.path.join(root_dir, f'batch_{batch_id:05d}.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rel_path in rel_paths:
            if rows_by_wsi is not None:
                row = rows_by_wsi.get(rel_path)
                if row is None:
                    location = source_csv or 'the provided slide list'
                    raise ValueError(f"Slide '{rel_path}' not found in {location}.")
                writer.writerow(row)
            else:
                writer.writerow({'wsi': rel_path})
    return csv_path


def initialize_processor(args: argparse.Namespace) -> Processor:
    """
    Initialize the Trident Processor with arguments set in `run_batch_of_slides`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing processor configuration.

    Returns
    -------
    Processor
        Initialized Trident Processor instance.
    """
    return Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        wsi_ext=args.wsi_ext,
        wsi_cache=args.wsi_cache,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
        custom_list_of_wsis=args.custom_list_of_wsis,
        max_workers=args.max_workers,
        reader_type=args.reader_type,
        search_nested=args.search_nested,
    )


def run_task(processor: Processor, args: argparse.Namespace) -> None:
    """
    Execute the specified task using the Trident Processor.

    Parameters
    ----------
    processor : Processor
        Initialized Trident Processor instance.
    args : argparse.Namespace
        Parsed command-line arguments containing task configuration.
    """

    if args.task == 'seg':
        from trident.segmentation_models.load import segmentation_model_factory

        # instantiate segmentation model and artifact remover in worker process to avoid pickle issues
        segmentation_model = segmentation_model_factory(
            args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        ).to(args.device)
        
        if args.remove_artifacts or args.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts
            ).to(args.device)
        else:
            artifact_remover_model = None

        # run segmentation 
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue= not args.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=args.seg_batch_size if args.seg_batch_size is not None else args.batch_size,
            device=args.device,
        )
    elif args.task == 'coords':
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=args.coords_dir,
            min_tissue_proportion=args.min_tissue_proportion
        )
    elif args.task == 'feat':
        if args.slide_encoder is None: 
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path).to(args.device)
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=args.device,
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.slide_encoder).to(args.device)
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                device=args.device,
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')


def worker_process(
    device: str,
    queue: "mp.queues.JoinableQueue[Optional[Tuple[int, str]]]",
    base_args: argparse.Namespace,
    task_sequence: Sequence[str],
) -> None:
    """Consume batches from the queue and execute tasks on a dedicated GPU."""

    threads = getattr(base_args, 'cpu_threads_per_worker', None)
    if threads is not None:
        try:
            torch.set_num_threads(threads)
        except (RuntimeError, AttributeError):
            pass
        try:
            torch.set_num_interop_threads(max(1, threads // 2) if threads > 1 else 1)
        except (RuntimeError, AttributeError):
            pass
        os.environ.setdefault('OMP_NUM_THREADS', str(threads))

    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_device(device)

    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break

        batch_id, location = item
        processor = None

        try:
            if base_args.wsi_cache:
                marker = os.path.join(location, '.cache_complete')
                while not os.path.exists(marker):
                    time.sleep(0.5)
                local_args = clone_args(base_args, wsi_dir=location, wsi_cache=None,
                                        custom_list_of_wsis=None, search_nested=False)
            else:
                local_args = clone_args(base_args, custom_list_of_wsis=location)

            print(f"[WORKER {device}] Processing batch {batch_id}.")
            processor = initialize_processor(local_args)

            for task_name in task_sequence:
                run_task(processor, clone_args(local_args, task=task_name, device=device))
        finally:
            if processor and hasattr(processor, 'release'):
                processor.release()
            if base_args.wsi_cache:
                shutil.rmtree(location, ignore_errors=True)
            elif location and os.path.exists(location):
                os.remove(location)
            queue.task_done()


def main() -> None:
    """
    Main entry point for the Trident batch processing script.
    
    Handles both sequential and parallel processing modes based on whether
    WSI caching is enabled. Supports segmentation, coordinate extraction,
    and feature extraction tasks.
    """
    # Parse arguments
    args = build_parser().parse_args()
    os.makedirs(args.job_dir, exist_ok=True)
    
    # Cleanup stale lock files and cache directory
    lock_count, cache_count = cleanup_files(args.job_dir, args.wsi_cache)
    if lock_count:
        print(f"[MAIN] Cleared {lock_count} stale lock file(s) under {args.job_dir}.")
    if cache_count:
        print(f"[MAIN] Cleared {cache_count} item(s) from cache directory {args.wsi_cache}.")
    
    devices = resolve_devices(args)
    task_sequence = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]

    list_workers = args.max_workers if args.max_workers and args.max_workers > 0 else 8
    if args.max_workers == 0:
        list_workers = 1
    full_paths, rel_paths = collect_valid_slides(
        wsi_dir=args.wsi_dir,
        custom_list_path=args.custom_list_of_wsis,
        wsi_ext=args.wsi_ext,
        search_nested=args.search_nested,
        max_workers=list_workers,
        return_relative_paths=True,
    )
    print(f"[MAIN] Found {len(full_paths)} valid slides in {args.wsi_dir}.")

    if not full_paths:
        print('[MAIN] No slides found. Exiting.')
        return

    rel_map = {full_path: rel_path for full_path, rel_path in zip(full_paths, rel_paths)}
    pending_paths = filter_completed_slides(full_paths, args, task_sequence)
    skipped = len(full_paths) - len(pending_paths)
    if skipped:
        print(f"[MAIN] Skipping {skipped} slide(s) with completed outputs.")

    if not pending_paths:
        print('[MAIN] All requested work already complete. Nothing to process.')
        return

    pending_infos = [(path, rel_map[path]) for path in pending_paths]
    
    # Determine batch size: cache mode uses cache_batch_size, non-cache mode splits evenly across GPUs
    if args.wsi_cache:
        batch_size = max(1, args.cache_batch_size or 32)
        os.makedirs(args.wsi_cache, exist_ok=True)
        print(f"[MAIN] Using cache directory {args.wsi_cache}.")
    else:
        batch_size = max(1, (len(pending_infos) + len(devices) - 1) // len(devices))
    
    csv_root = None if args.wsi_cache else os.path.join(args.job_dir, '_trident_batches')

    if args.custom_list_of_wsis:
        custom_rows, custom_columns = load_custom_slide_rows(args.custom_list_of_wsis)
    else:
        custom_rows, custom_columns = None, ['wsi']

    ctx = mp.get_context("spawn")
    queue = ctx.JoinableQueue(maxsize=2)  # Pre-cache 2 batches

    def cacher() -> None:
        """Pre-caches batches and puts ready paths in queue."""
        for batch_id, start_idx in enumerate(range(0, len(pending_infos), batch_size)):
            batch_infos = pending_infos[start_idx:start_idx + batch_size]
            
            if args.wsi_cache:
                full_batch = [info[0] for info in batch_infos]
                dest_dir = os.path.join(args.wsi_cache, f'batch_{batch_id}')
                print(f"[CACHE] Caching batch {batch_id} to {dest_dir}.")
                cache_batch(full_batch, dest_dir)
                queue.put((batch_id, dest_dir))
            else:
                rel_batch = [info[1] for info in batch_infos]
                csv_path = write_batch_csv(rel_batch, batch_id, csv_root, 
                                          custom_rows, custom_columns, args.custom_list_of_wsis)
                queue.put((batch_id, csv_path))
        
        for _ in devices:
            queue.put(None)

    # Start cacher thread
    Thread(target=cacher, daemon=True).start()

    workers = [ctx.Process(target=worker_process, args=(device, queue, args, task_sequence))
               for device in devices]
    for process in workers:
        process.start()

    print(f"[MAIN] Dispatching {len(pending_infos)} slide(s) across {len(devices)} device(s).")
    queue.join()
    for process in workers:
        process.join()

    # Cleanup temporary batch CSV directory
    if csv_root and os.path.isdir(csv_root):
        shutil.rmtree(csv_root, ignore_errors=True)


if __name__ == "__main__":
    main()
