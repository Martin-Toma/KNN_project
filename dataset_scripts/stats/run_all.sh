. ../../venv/bin/activate

# Run every .py file in the current directory
for file in *.py; do
    echo "Running $file"
    python "$file" > "${file%.py}.log" 2>&1
    echo "-----------------------------------------------------------"
done