# Run the processing and save in a zip.
#
echo "Processing data..."
formats=(
    Standard
    Pioneer
    Modern
)

for i in "${formats[@]}"; do
    echo "running process_data.py $i"
    python process_data.py $i
    echo "zipping"
    zip $i.zip processed_data/*
    rm processed_data/*
done

echo "Last updated: $(date)" > data_version.txt