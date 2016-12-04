rm -rf *.pkl
rm -rf videos/frames
rm -rf videos/gradients
mkdir -p videos/frames
mkdir -p videos/gradients
ffmpeg -i $1 -r 25 videos/frames/output_%04d.png
cd matlab
./run_matlab.sh compute_gradient
cd ..
python main.py
