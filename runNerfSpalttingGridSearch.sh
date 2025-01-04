BASE_CONFIG="./gnerf/config/synthetic_occ.yaml"
CONFIG_DIR="./gnerf/config"
SBATCH_DIR="./gnerf_sbatch_jobs"

SBATCH_TEMPLATE="#!/bin/bash
#SBATCH --job-name=grid-search-nerf-splatting
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=student

source activate nerf-splatting
cd gnerf
"

mkdir -p "$SBATCH_DIR"

N_NEIGHBOURS=(16 32 64)
N_FEATURES_PER_GAUSS=(32 64 128)
N_GAUSSES=(75000 100000)

for NEIGHBOURS in "${N_NEIGHBOURS[@]}"; do
  for FEATURES in "${N_FEATURES_PER_GAUSS[@]}"; do
    for GAUSSES in "${N_GAUSSES[@]}"; do
      # Create unique filenames
      CONFIG_FILE="synthetic_occ_neighbours_${NEIGHBOURS}_features_${FEATURES}_gausses_${GAUSSES}.yaml"
      CONFIG_FILE_FULL="$CONFIG_DIR/$CONFIG_FILE"
      SBATCH_FILE="$SBATCH_DIR/sbatch_neighbours_${NEIGHBOURS}_features_${FEATURES}_gausses_${GAUSSES}.sh"

      # Generate YAML configuration
      sed -e "s/^.*n_neighbours:.*$/    n_neighbours: $NEIGHBOURS/" \
          -e "s/^.*n_features_per_gauss:.*$/    n_features_per_gauss: $FEATURES/" \
          -e "s/^.*n_gausses:.*$/    n_gausses: $GAUSSES/" \
          "$BASE_CONFIG" > "$CONFIG_FILE_FULL"

      # Create SBATCH script
      echo "$SBATCH_TEMPLATE" > "$SBATCH_FILE"
      echo "python examples/train_laghash_nerf_occ.py --config-name \"$CONFIG_FILE\" dataset.scene=\"chair\" dataset.data_root=\"/shared/sets/datasets/nerf_synthetic\"" >> "$SBATCH_FILE"
      
      chmod +x "$SBATCH_FILE"
      # Submit the job
      echo "$SBATCH_FILE"
      sbatch "$SBATCH_FILE"
    done
  done
done