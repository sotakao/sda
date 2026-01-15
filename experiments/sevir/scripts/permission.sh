#!/bin/bash
DESTINATION_DIR="/proj/berzelius-2022-164/weather/SQG/experiments/results"

# Fix permissions after training
echo "Fixing permissions..."
find "$DESTINATION_DIR" -name "*.nc" -type f -exec chmod 660 {} \;
find "$DESTINATION_DIR" -type d -exec chmod 775 {} \;
echo "Done!"