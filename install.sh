# Create the activation directory if it doesn't exist
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# Create a script that will run when the environment is activated
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash

# Get path to the conda environment
CONDA_ENV_PATH=$(conda info --base)/envs/$(basename "$CONDA_PREFIX")

# Find CUDA toolkit path (adjust version if needed)
CUDA_HOME=$(find "$CONDA_ENV_PATH" -type d -path "*cudatoolkit*" | grep -E "cuda|cudatoolkit" | head -n 1)

# Fallback if above fails
if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME="$CONDA_ENV_PATH"
fi

export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$(find $CONDA_PREFIX/lib/python3.9/site-packages/nvidia -type d -name include | paste -sd: -):$CPLUS_INCLUDE_PATH

export LDFLAGS="-L$CONDA_PREFIX/lib"

echo "CUDA_HOME set to $CUDA_HOME"
echo "LD_LIBRARY_PATH set to $LD_LIBRARY_PATH"
EOF

# Make the script executable
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Create a deactivation script to clean up the environment variables
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/bash

# Unset environment variables
unset CUDA_HOME
unset LDFLAGS

# Note: We don't unset LD_LIBRARY_PATH, LIBRARY_PATH, or CPLUS_INCLUDE_PATH completely
# as they might have had values before activation. Instead, you might want to
# save their original values in the activate script and restore them here.
EOF

# Make the deactivation script executable
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

echo "Environment variables have been set up to be automatically configured when activating this Conda environment."


# Get path to the conda environment
CONDA_ENV_PATH=$(conda info --base)/envs/$(basename "$CONDA_PREFIX")

# Find CUDA toolkit path (adjust version if needed)
CUDA_HOME=$(find "$CONDA_ENV_PATH" -type d -path "*cudatoolkit*" | grep -E "cuda|cudatoolkit" | head -n 1)

# Fallback if above fails
if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME="$CONDA_ENV_PATH"
fi

export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$(find $CONDA_PREFIX/lib/python3.9/site-packages/nvidia -type d -name include | paste -sd: -):$CPLUS_INCLUDE_PATH

export LDFLAGS="-L$CONDA_PREFIX/lib"

echo "CUDA_HOME set to $CUDA_HOME"
echo "LD_LIBRARY_PATH set to $LD_LIBRARY_PATH"