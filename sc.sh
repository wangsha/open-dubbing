# Script that mimics Softcatalà setup
rm -f open_dubbing.log
pip install .
rm -r -f output/
branch_name=$(git rev-parse --abbrev-ref HEAD)

declare -a target_languages=("cat")  # Catalan (cat) and French (fra)
declare -a inputs=($(find ../dubbing/od-videos/ -type f -name "*.mp4"))
declare -a inputs=("videos/jordi.mp4" )

for input_file in "${inputs[@]}"; do
  output_directory="output/$(basename "${input_file%.*}").${branch_name}/"
  for language in "${target_languages[@]}"; do

    # Run the dubbing command
    open-dubbing \
      --input_file "$input_file" \
      --whisper_model="medium" \
      --vad \
      --output_directory="$output_directory" \
      --target_language="$language" \
      --translator="apertium" \
      --apertium_server=http://localhost:8500/ \
      --tts=mms \
      --device=cpu \
      --dubbed_subtitles\
      --log_level=INFO
    if [ $? -ne 0 ]; then
        echo "Error occurred with open-dubbing. Exiting loop."
        exit 1
    fi
    echo ""
  done
done

