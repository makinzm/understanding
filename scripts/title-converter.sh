#!/bin/bash

# Convert title to filename
# Usage: bash title-converter.sh
#        echo "Title" | bash title-converter.sh (for testing)

convert_title() {
    local title="$1"
    echo "$title" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
}

if [ -t 0 ]; then
    # Interactive mode
    echo -n "Enter title: "
    read title
    filename=$(convert_title "$title").md
    echo "$filename"
else
    # Pipe mode for testing
    read title
    filename=$(convert_title "$title").md
    echo "$filename"
fi
