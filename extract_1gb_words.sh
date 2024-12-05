#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

input_file="$1"
output_file="$2"

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi


target_size=$((1024 * 1024 * 1024)) # 1GB in bytes


sample_size=10000

# We'll take three samples:
# 1. From the start of the file
# 2. From the middle of the file
# 3. From near the end of the file
total_bytes=$(wc -c < "$input_file")
if [ "$total_bytes" -eq 0 ]; then
    echo "Error: Input file is empty."
    exit 1
fi

offset_mid=$((total_bytes / 2))
offset_end=$(( (total_bytes * 3) / 4 )) # 3/4 into the file

echo "Estimating average word size..."

# Function to measure average word size from a given offset
measure_avg_word_size() {
    local offset="$1"
    local tmpfile
    tmpfile=$(mktemp)
    # Extract a chunk starting at 'offset' to get ~ sample_size words
    # We'll read sample_size*10 bytes just as a heuristic
    dd if="$input_file" bs=1 skip="$offset" count=$((sample_size * 10)) 2>/dev/null \
        | awk '{for(i=1; i<=NF; i++) print $i}' \
        | head -n "$sample_size" > "$tmpfile"

    local bytes
    bytes=$(wc -c < "$tmpfile")

    local wordcount
    wordcount=$(wc -l < "$tmpfile")
    if [ "$wordcount" -eq 0 ]; then
        # If we couldn't get any words, return 0 to indicate failure
        rm "$tmpfile"
        echo 0
        return
    fi

    # Compute average word size as floating-point
    # We'll just print the floating result and let caller handle it
    awk -v total_bytes="$bytes" -v total_words="$wordcount" 'BEGIN {print total_bytes / total_words}' "$tmpfile"

    rm "$tmpfile"
}

# Measure three samples
avg_word_size_start=$(measure_avg_word_size 0)
avg_word_size_mid=$(measure_avg_word_size "$offset_mid")
avg_word_size_end=$(measure_avg_word_size "$offset_end")

# Filter out any zeros (failed samples)
samples=()
[ "$(printf '%.0f' "$avg_word_size_start")" -gt 0 ] && samples+=("$avg_word_size_start")
[ "$(printf '%.0f' "$avg_word_size_mid")" -gt 0 ] && samples+=("$avg_word_size_mid")
[ "$(printf '%.0f' "$avg_word_size_end")" -gt 0 ] && samples+=("$avg_word_size_end")

if [ "${#samples[@]}" -eq 0 ]; then
    echo "Error: Unable to sample words from file."
    exit 1
fi

# Average the samples
sum=0
for s in "${samples[@]}"; do
    sum=$(awk -v a="$sum" -v b="$s" 'BEGIN {print a+b}')
done
avg_word_size=$(awk -v s="$sum" -v c="${#samples[@]}" 'BEGIN {print s/c}')

# Compute estimated number of words
estimated_words=$(awk -v tgt="$target_size" -v avg="$avg_word_size" 'BEGIN {
    if (avg > 0) print int(tgt / avg); else print 0
}')

echo "Estimated words for 1GB: $estimated_words"
echo "Extracting words..."

awk -v max_words="$estimated_words" '
{
    for (i=1; i<=NF; i++) {
        count++
        if (count <= max_words) {
            print $i
        } else {
            exit
        }
    }
}' "$input_file" > "$output_file"

echo "Extraction complete. Output saved to '$output_file'."
